from datetime import date
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quantv1.ingest.earnings import PROTOCOL_LOCK_DATE, RETROSPECTIVE_HOLDOUT_START
from quantv1.portfolio.ledger import PortfolioLedger
from quantv1.research.earnings_alpha import (
    EarningsStudyError, _load_feature_metadata, build_feature_frame,
)
from quantv1.research.protocol import (
    BETA_VERSION, TARGET_VERSION, clustered_mean_ci,
    clustered_portfolio_bootstrap, execution_cost_estimate,
    first_session_after, hac_statistics, power_requirements,
    shrink_and_clip_beta, trading_sessions,
)


class ProtocolCorrectionTests(unittest.TestCase):
    def test_frozen_dates_and_exchange_calendar(self):
        self.assertEqual(RETROSPECTIVE_HOLDOUT_START, date(2025, 7, 1))
        self.assertEqual(PROTOCOL_LOCK_DATE, date(2026, 7, 10))
        self.assertEqual(first_session_after("2026-07-10T23:00:00Z"), "2026-07-13")
        self.assertEqual(trading_sessions("2026-04-02", "2026-04-06"),
                         [date(2026, 4, 2), date(2026, 4, 6)])

    def test_beta_is_shrunk_and_clipped(self):
        self.assertEqual(shrink_and_clip_beta(float("nan"), 0), 1.0)
        self.assertAlmostEqual(shrink_and_clip_beta(2.0, 20), 1.5)
        self.assertEqual(shrink_and_clip_beta(10.0, 1000), 2.0)
        self.assertEqual(shrink_and_clip_beta(-10.0, 1000), 0.0)
        self.assertIn("pre-event", BETA_VERSION)
        self.assertIn("beta-hedged", TARGET_VERSION)

    def test_short_costs_fail_closed_without_historical_borrow(self):
        base = {"beta": 1.0, "quote_complete": False}
        self.assertFalse(execution_cost_estimate(
            {"beta": None}, 1
        )["deployable"])
        self.assertFalse(execution_cost_estimate(base, -1)["deployable"])
        short = execution_cost_estimate(
            {**base, "borrow_available": True, "borrow_fee_bps_annual": 252.0}, -1
        )
        self.assertTrue(short["deployable"])
        self.assertGreater(short["all_in_cost"],
                           execution_cost_estimate(base, 1)["all_in_cost"])

    def test_observed_spread_and_liquidity_drive_quote_cost(self):
        row = {"beta": 1.0, "quote_complete": True, "trailing_adv": 10_000_000,
               "entry_bid": 99.9, "entry_ask": 100.1,
               "exit_bid": 100.9, "exit_ask": 101.1,
               "benchmark_entry_bid": 49.95, "benchmark_entry_ask": 50.05,
               "benchmark_exit_bid": 50.45, "benchmark_exit_ask": 50.55}
        liquid = execution_cost_estimate(row, 1)
        illiquid = execution_cost_estimate({**row, "trailing_adv": 100_000}, 1)
        self.assertEqual(liquid["mode"], "NBBO")
        self.assertGreater(illiquid["all_in_cost"], liquid["all_in_cost"])

    def test_calendar_keeps_zero_return_cash_sessions(self):
        calendar = [date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)]
        result = PortfolioLedger(calendar=calendar).run([])
        self.assertEqual(result["daily_returns"], [0.0, 0.0, 0.0])
        self.assertEqual([row["date"] for row in result["nav_path"]],
                         [str(day) for day in calendar])

    def test_hac_bootstrap_and_power_are_frozen(self):
        hac = hac_statistics([0.01, -0.002, 0.003, 0.0, 0.004, 0.001])
        self.assertEqual((hac["lag"], hac["n_sessions"]), (5, 6))
        power = power_requirements(0.04)
        self.assertEqual(power["status"], "FROZEN")
        self.assertGreater(power["minimum_unique_executable_trades"], 0)
        frame = pd.DataFrame({"ticker": ["A", "B", "A", "B"],
                              "announcement_date": ["2024-01-01", "2024-01-01",
                                                    "2024-02-01", "2024-02-01"]})
        self.assertEqual(clustered_mean_ci(
            frame, [0.01, 0.02, 0.03, 0.04], draws=50
        )["status"], "COMPLETE")
        portfolio = {
            "trades": [{"earnings_event_id": str(i), "ticker": ticker,
                        "announcement_date": day}
                       for i, (ticker, day) in enumerate(zip(
                           frame.ticker, frame.announcement_date))],
            "nav_path": [{"date": "2024-01-02"}, {"date": "2024-01-03"}],
            "trade_daily_pnl": [{"earnings_event_id": str(i),
                                 "date": "2024-01-02", "pnl": 0.001 * (i + 1)}
                                for i in range(4)],
        }
        boot = clustered_portfolio_bootstrap(portfolio, draws=50)
        self.assertEqual(boot["status"], "COMPLETE")
        self.assertIn("sharpe_annual", boot["confidence_intervals"])

    def test_holdout_feature_build_requires_model_lock_and_full_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with patch("quantv1.research.earnings_alpha.SPEC_LOCK_PATH", missing):
                with self.assertRaises(EarningsStudyError):
                    build_feature_frame(mode="full", before=date(2026, 1, 1))
            lock = Path(directory) / "lock.json"
            lock.write_text("{}")
            with patch("quantv1.research.earnings_alpha.SPEC_LOCK_PATH", lock):
                with self.assertRaises(EarningsStudyError):
                    build_feature_frame(mode="coarse", before=date(2026, 1, 1))

    def test_only_full_current_artifact_is_promotion_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "metadata.json"
            with patch("quantv1.research.earnings_alpha.FEATURE_METADATA_PATH", metadata):
                metadata.write_text('{"mode":"coarse","target_version":"%s",'
                                    '"beta_version":"%s"}' %
                                    (TARGET_VERSION, BETA_VERSION))
                self.assertFalse(_load_feature_metadata()["promotion_eligible"])
                metadata.write_text('{"mode":"full","target_version":"%s",'
                                    '"beta_version":"%s"}' %
                                    (TARGET_VERSION, BETA_VERSION))
                self.assertTrue(_load_feature_metadata()["promotion_eligible"])


if __name__ == "__main__":
    unittest.main()
