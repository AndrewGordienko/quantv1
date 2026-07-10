from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from quantv1.db import connect
from quantv1.ingest import earnings
from quantv1.research.earnings_alpha import (
    _execution_prices, _five_day_exit, _known_bar_close, _next_bar_open,
    purged_group_time_splits, simulate_portfolio,
)
from quantv1.v4.earnings_windows import window_bounds


class EarningsIngestTests(unittest.TestCase):
    def test_sec_filing_classifier_requires_results_financials_and_period(self):
        filing = """
        <DOCUMENT><TYPE>8-K<TEXT>
        The company issued a press release reporting financial results for the
        fiscal quarter ended March 31, 2025. Revenue increased, net income rose,
        and earnings per diluted share were $1.20.
        </TEXT></DOCUMENT>
        <DOCUMENT><TYPE>EX-99.1<FILENAME>release.htm<TEXT>
        Quarterly financial results. Consolidated statements of operations.
        </TEXT></DOCUMENT>
        """
        result = earnings.classify_sec_filing_text(
            filing, datetime(2025, 4, 20, 20, 0)
        )
        self.assertEqual(result["event_classification"],
                         "VERIFIED_EARNINGS_RELEASE")
        self.assertEqual(result["fiscal_period_end"], "2025-03-31")
        self.assertEqual(result["timestamp_quality"], "TIER_2_SEC_ACCEPTANCE")
        self.assertEqual(result["exhibit_filename"], "release.htm")

    def test_sec_filing_classifier_rejects_operational_item_202(self):
        filing = """
        <DOCUMENT><TYPE>8-K<TEXT>
        The company reports quarterly vehicle production and deliveries.
        </TEXT></DOCUMENT>
        <DOCUMENT><TYPE>EX-99.1<FILENAME>deliveries.htm<TEXT>
        Vehicles produced and delivered for the quarter.
        </TEXT></DOCUMENT>
        """
        result = earnings.classify_sec_filing_text(
            filing, datetime(2025, 4, 2, 12, 0)
        )
        self.assertEqual(result["event_classification"], "NOT_EARNINGS")

    def test_item_202_is_unverified_until_direct_release(self):
        record = {
            "form": "8-K", "items": "2.02,9.01", "filingDate": "2025-01-29",
            "acceptanceDateTime": "2025-01-29T16:05:00-05:00",
            "reportDate": "2025-01-29", "accessionNumber": "0001-25-000001",
            "primaryDocument": "issuer-8k.htm",
        }
        source = earnings._sec_source("TEST", "123", record)
        self.assertIsNotNone(source)
        metadata = json.loads(source[-1])
        self.assertEqual(metadata["event_classification"],
                         "UNVERIFIED_EARNINGS_CANDIDATE")
        self.assertIsNone(source[3])  # fiscal period is not inferred from filing date

    def test_direct_release_promotes_and_consensus_is_point_in_time_only(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.duckdb"
            release_path = Path(directory) / "release.json"
            consensus_path = Path(directory) / "consensus.json"
            with patch("quantv1.db.DB_PATH", db_path):
                connect().close()
                release_path.write_text(json.dumps([{
                    "ticker": "TEST", "cik": "123", "fiscal_period_end": "2024-12-31",
                    "public_time": "2025-01-29T16:01:00-05:00",
                    "source_type": "company_ir", "source_url": "https://ir.test.com/q4",
                    "reviewed_earliest": True,
                }]))
                result = earnings.ingest_release_manifest(release_path)
                self.assertEqual(result["events_affected"], 1)
                con = connect(read_only=True)
                event = con.execute("""
                    SELECT earnings_event_id,timestamp_status,release_session
                    FROM earnings_events
                """).fetchone()
                con.close()
                self.assertEqual(event[1:], ("VERIFIED_EARLIEST", "AMC"))

                consensus_path.write_text(json.dumps([{
                    "earnings_event_id": event[0], "metric": "diluted_eps",
                    "estimate_value": 1.2, "estimate_as_of": "2025-01-29T15:00:00-05:00",
                    "vendor": "licensed", "vendor_record_id": "r1",
                    "is_point_in_time": True, "is_final_revised": False,
                }]))
                self.assertEqual(earnings.ingest_consensus_manifest(consensus_path)
                                 ["consensus_snapshots"], 1)
                bad = json.loads(consensus_path.read_text())
                bad[0]["vendor_record_id"] = "r2"
                bad[0]["is_final_revised"] = True
                consensus_path.write_text(json.dumps(bad))
                with self.assertRaises(earnings.EarningsDataError):
                    earnings.ingest_consensus_manifest(consensus_path)

    def test_release_time_requires_offset_and_review(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.duckdb"
            path = Path(directory) / "release.json"
            path.write_text(json.dumps([{
                "ticker": "TEST", "fiscal_period_end": "2024-12-31",
                "public_time": "2025-01-29T16:01:00", "source_type": "company_ir",
                "source_url": "https://ir.test.com/q4", "reviewed_earliest": False,
            }]))
            with patch("quantv1.db.DB_PATH", db_path):
                connect().close()
                with self.assertRaises(earnings.EarningsDataError):
                    earnings.ingest_release_manifest(path)

    def test_company_size_must_be_known_at_universe_formation(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.duckdb"
            path = Path(directory) / "size.json"
            with patch("quantv1.db.DB_PATH", db_path):
                con = connect()
                con.execute("""
                    INSERT INTO earnings_universe_snapshots
                        (universe_version,ticker,eligibility_as_of,company_bucket,
                         included,first_seen_at)
                    VALUES (?,?,?,'TRAIN_COMPANY',TRUE,?)
                """, [earnings.UNIVERSE_VERSION, "TEST", "2021-06-30",
                      datetime(2021, 6, 30)])
                con.close()
                record = {"ticker": "TEST", "market_cap": 12_000_000_000,
                          "known_at": "2021-07-01T00:00:00Z", "source": "vendor",
                          "source_record_id": "size-1"}
                path.write_text(json.dumps([record]))
                with self.assertRaises(earnings.EarningsDataError):
                    earnings.ingest_universe_metadata_manifest(path)
                record["known_at"] = "2021-06-29T20:00:00Z"
                path.write_text(json.dumps([record]))
                earnings.ingest_universe_metadata_manifest(path)
                con = connect(read_only=True)
                bucket = con.execute("""
                    SELECT company_size_bucket FROM earnings_universe_snapshots
                    WHERE ticker='TEST'
                """).fetchone()[0]
                con.close()
                self.assertEqual(bucket, "large")


class EarningsExecutionTests(unittest.TestCase):
    def test_cv_is_time_purged_and_company_grouped(self):
        frame = pd.DataFrame({
            "entry_time": pd.date_range("2021-01-01", periods=240, freq="D", tz="UTC"),
            "ticker": [f"T{i % 20}" for i in range(240)],
            "earnings_event_id": [f"E{i}" for i in range(240)],
        })
        for training, validation in purged_group_time_splits(frame, 3):
            self.assertTrue(set(frame.iloc[training]["earnings_event_id"]).isdisjoint(
                set(frame.iloc[validation]["earnings_event_id"])))
            self.assertLess(frame.iloc[training]["entry_time"].max(),
                            frame.iloc[validation]["entry_time"].min() -
                            pd.Timedelta(days=20))

    def test_known_close_does_not_use_current_bar_close(self):
        timestamps = pd.date_range("2025-01-02 14:30", periods=3, freq="1min")
        bars = pd.DataFrame({"ts": timestamps, "close": [100.0, 200.0, 300.0]})
        # At 14:31 the 14:31 bar has only just opened; only 14:30 close is known.
        self.assertEqual(_known_bar_close(bars, pd.Timestamp("2025-01-02 14:31", tz="UTC")),
                         100.0)
        entry = _next_bar_open(bars.assign(open=[99.0, 199.0, 299.0]),
                               pd.Timestamp("2025-01-02 14:31", tz="UTC"))
        self.assertEqual(entry["price"], 199.0)

    def test_five_day_exit_is_fifth_subsequent_common_close(self):
        dates = pd.bdate_range("2025-01-02", periods=7)
        timestamps = [pd.Timestamp(f"{day.date()} 20:59", tz="UTC") for day in dates]
        asset = pd.DataFrame({"ts": timestamps, "open": [100] * 7,
                              "close": [100 + index for index in range(7)]})
        benchmark = pd.DataFrame({"ts": timestamps, "open": [50] * 7,
                                  "close": [50 + index for index in range(7)]})
        result = _five_day_exit(asset, benchmark, dates[0].date())
        self.assertEqual(result["date"], dates[5].date())
        self.assertEqual(result["asset_close"], 105.0)

    def test_amc_window_extends_through_next_session(self):
        public = datetime(2025, 1, 3, 21, 5)  # Friday 16:05 ET
        start, end = window_bounds(public, "AMC")
        self.assertLess(start, public)
        self.assertGreaterEqual(end.date(), datetime(2025, 1, 6).date())

    def test_portfolio_respects_overlapping_gross_and_bar_costs(self):
        rows = []
        base = pd.Timestamp("2025-01-02 14:35", tz="UTC")
        for index in range(8):
            rows.append({
                "earnings_event_id": f"e{index}", "ticker": f"T{index}",
                "sector": "Technology", "release_session": "BMO",
                "entry_time": base, "exit_time": base + pd.Timedelta(hours=2),
                "delayed_entry_time": base + pd.Timedelta(minutes=10),
                "delayed_exit_time": base + pd.Timedelta(hours=2, minutes=10),
                "entry_price": 100.0, "exit_price": 101.0,
                "delayed_entry_price": 100.1,
                "benchmark_entry_price": 100.0,
                "benchmark_delayed_entry_price": 100.0,
                "benchmark_exit_price": 100.0, "beta": 1.0,
            })
        result = simulate_portfolio(pd.DataFrame(rows), np.full(8, 0.02))
        # Each beta-one trade consumes 10% gross (5% asset + 5% ETF hedge).
        self.assertEqual(result["n_trades"], 2)
        self.assertGreater(result["net_return"], 0)
        self.assertGreater(result["hedge_turnover"], 0)

    def test_quote_execution_crosses_both_stock_and_hedge_spreads(self):
        row = {
            "beta": 1.0, "quote_complete": True,
            "entry_bid": 99.0, "entry_ask": 101.0,
            "delayed_entry_bid": 98.0, "delayed_entry_ask": 102.0,
            "exit_bid": 109.0, "exit_ask": 111.0,
            "benchmark_entry_bid": 49.0, "benchmark_entry_ask": 51.0,
            "benchmark_delayed_entry_bid": 48.0,
            "benchmark_delayed_entry_ask": 52.0,
            "benchmark_exit_bid": 49.0, "benchmark_exit_ask": 51.0,
            "entry_price": 100.0, "delayed_entry_price": 100.0,
            "exit_price": 110.0, "benchmark_entry_price": 50.0,
            "benchmark_delayed_entry_price": 50.0,
            "benchmark_exit_price": 50.0,
        }
        long_prices = _execution_prices(row, 1)
        self.assertEqual(long_prices["mode"], "NBBO")
        self.assertEqual((long_prices["entry"], long_prices["exit"]),
                         (101.0, 109.0))
        self.assertEqual((long_prices["benchmark_entry"],
                          long_prices["benchmark_exit"]), (49.0, 51.0))
        short_prices = _execution_prices(row, -1, delayed=True)
        self.assertEqual((short_prices["entry"], short_prices["exit"]),
                         (98.0, 111.0))
        self.assertEqual((short_prices["benchmark_entry"],
                          short_prices["benchmark_exit"]), (52.0, 49.0))


if __name__ == "__main__":
    unittest.main()
