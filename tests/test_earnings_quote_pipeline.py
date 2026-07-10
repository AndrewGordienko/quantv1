from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

from quantv1.research.earnings_alpha import _window_features, simulate_portfolio


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    con = duckdb.connect(":memory:")
    try:
        con.register("fixture", frame)
        con.execute(f"COPY fixture TO '{path}' (FORMAT PARQUET)")
    finally:
        con.close()


def _bars(multiplier: float) -> pd.DataFrame:
    records = []
    for day_index, day in enumerate(pd.bdate_range("2025-01-02", periods=7)):
        price = multiplier + day_index
        for clock in ("14:30", "14:31", "14:35", "15:00", "15:30", "20:59"):
            records.append({
                "ts": pd.Timestamp(f"{day.date()} {clock}", tz="UTC"),
                "open": price, "high": price + 0.1, "low": price - 0.1,
                "close": price, "volume": 10_000.0,
            })
    return pd.DataFrame(records)


def _quotes(multiplier: float, *, stale_entry: bool = False,
            missing_exit: bool = False) -> pd.DataFrame:
    entry_clock = "15:03" if stale_entry else "15:00"
    records = [
        {"ts": pd.Timestamp(f"2025-01-02 {entry_clock}", tz="UTC"),
         "bid": multiplier - 0.1, "ask": multiplier + 0.1},
        {"ts": pd.Timestamp("2025-01-02 15:30", tz="UTC"),
         "bid": multiplier, "ask": multiplier + 0.2},
    ]
    if not missing_exit:
        records.append({
            "ts": pd.Timestamp("2025-01-09 20:59", tz="UTC"),
            "bid": multiplier + 4.9, "ask": multiplier + 5.1,
        })
    return pd.DataFrame(records)


class HistoricalQuotePipelineTests(unittest.TestCase):
    def _feature(self, directory: str, *, stale_asset_entry: bool = False,
                 missing_benchmark_exit: bool = False) -> dict:
        root = Path(directory)
        paths = {
            "bars": root / "asset_bars.parquet",
            "benchmark_bars": root / "benchmark_bars.parquet",
            "quotes": root / ("asset_quotes_stale.parquet" if stale_asset_entry
                              else "asset_quotes.parquet"),
            "benchmark_quotes": root / (
                "benchmark_quotes_missing.parquet" if missing_benchmark_exit
                else "benchmark_quotes.parquet"
            ),
        }
        _write_parquet(paths["bars"], _bars(100.0))
        _write_parquet(paths["benchmark_bars"], _bars(50.0))
        _write_parquet(paths["quotes"], _quotes(
            100.0, stale_entry=stale_asset_entry
        ))
        _write_parquet(paths["benchmark_quotes"], _quotes(
            50.0, missing_exit=missing_benchmark_exit
        ))
        row = SimpleNamespace(
            earnings_event_id="historical-quote-fixture", ticker="TEST",
            earliest_public_time=datetime(2025, 1, 2, 13, 0,
                                          tzinfo=timezone.utc),
            release_session="BMO", timestamp_status="VERIFIED_EARLIEST",
            fiscal_quarter="Q4", sector="Technology", benchmark_ticker="XLK",
            company_bucket="TRAIN_COMPANY", company_size_bucket="large",
            quote_coverage=1.0,
            bars_path=str(paths["bars"]), quotes_path=str(paths["quotes"]),
            benchmark_bars_path=str(paths["benchmark_bars"]),
            benchmark_quotes_path=str(paths["benchmark_quotes"]),
        )
        context = {
            "prior_close": 99.0, "pre_event_volatility": 0.02,
            "trailing_adv": 100_000_000.0, "beta": 1.0,
        }
        with patch("quantv1.research.earnings_alpha._daily_context",
                   return_value=context), patch(
                       "quantv1.research.earnings_alpha._consensus_actuals",
                       return_value={}):
            feature = _window_features(None, row)
        self.assertIsNotNone(feature)
        return feature

    def test_feature_to_nbbo_execution_to_daily_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            feature = self._feature(directory)
        self.assertTrue(feature["quote_complete"])
        self.assertEqual(feature["execution_mode"],
                         "NEXT_EXECUTABLE_NBBO_PLUS_ASSUMED_COST")
        result = simulate_portfolio(pd.DataFrame([feature]), np.asarray([0.02]))
        self.assertEqual(result["n_trades"], 1)
        trade = result["trades"][0]
        self.assertEqual(trade["execution_mode"], "NBBO")
        self.assertAlmostEqual(trade["entry_price"], 100.1)
        self.assertAlmostEqual(trade["benchmark_entry_price"], 49.9)
        self.assertEqual(len(result["nav_path"]), 6)
        self.assertGreater(result["hedge_turnover"], 0)

    def test_stale_or_missing_quote_disables_nbbo_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            feature = self._feature(
                directory, stale_asset_entry=True,
                missing_benchmark_exit=True,
            )
        self.assertFalse(feature["quote_complete"])
        result = simulate_portfolio(pd.DataFrame([feature]), np.asarray([0.02]))
        self.assertEqual(result["n_trades"], 1)
        self.assertEqual(result["trades"][0]["execution_mode"], "BAR")


if __name__ == "__main__":
    unittest.main()
