from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from quantv1.portfolio.ledger import PortfolioLedger
from quantv1.research.earnings_alpha import (
    PRICE_CATEGORICAL,
    PRICE_NUMERIC,
    _available_model_specs,
    add_structured_features,
    calculate_fundamental_surprise,
    calculate_mismatch,
    coverage_statistics,
    permutation_controls,
)
from quantv1.research.earnings_strategy import decision_from_prediction


def _coverage_frame() -> pd.DataFrame:
    rows = []
    for split, year in (("TRAIN_TIME", 2023), ("VALIDATION_TIME", 2025)):
        for index in range(40):
            rows.append({
                "time_bucket": split,
                "entry_time": pd.Timestamp(f"{year}-01-02", tz="UTC") +
                              pd.Timedelta(days=index),
                "sector": "Technology" if index < 20 else "Healthcare",
                "company_size_bucket": "large" if index % 2 else "mid",
                "has_point_in_time_consensus": 1.0,
                "representative_consensus_coverage": 1.0,
            })
    return pd.DataFrame(rows)


class StructuredResearchProtocolTests(unittest.TestCase):
    def test_structured_gate_requires_80_percent_in_both_splits(self):
        frame = _coverage_frame()
        frame.loc[frame.time_bucket.eq("VALIDATION_TIME") &
                  frame.index.to_series().mod(5).eq(0),
                  "representative_consensus_coverage"] = 0.0
        self.assertTrue(coverage_statistics(frame)["gate_passed"])
        validation_index = frame.index[frame.time_bucket.eq("VALIDATION_TIME")][1]
        frame.loc[validation_index, "representative_consensus_coverage"] = 0.0
        result = coverage_statistics(frame)
        self.assertFalse(result["gate_passed"])
        self.assertLess(result["by_split"]["VALIDATION_TIME"]["coverage"], 0.80)

    def test_coverage_gate_requires_company_size_representation(self):
        frame = _coverage_frame().drop(columns="company_size_bucket")
        result = coverage_statistics(frame)
        self.assertFalse(result["gate_passed"])
        self.assertIn("company_size_bucket", result["missing_dimensions"])

    def test_structured_scalers_fit_training_only(self):
        base = pd.DataFrame({
            "time_bucket": ["TRAIN_TIME"] * 4 + ["VALIDATION_TIME"],
            "entry_time": pd.date_range("2023-01-01", periods=5, tz="UTC"),
            "sector": ["Technology"] * 5,
            "company_size_bucket": ["large"] * 5,
            "eps_surprise_raw": [0.0, 1.0, 2.0, 3.0, 1_000.0],
            "revenue_surprise_raw": [0.0, 1.0, 2.0, 3.0, 1_000.0],
            "guidance_surprise_raw": [0.0, 1.0, 2.0, 3.0, 1_000.0],
            "analyst_dispersion_raw": [0.0, 1.0, 2.0, 3.0, 1_000.0],
            "revision_breadth": [0.0] * 5,
            "reaction_score": [0.0] * 5,
            "trailing_adv": [10_000_000.0] * 5,
            "has_point_in_time_consensus": [1.0] * 5,
        })
        first, first_stats = add_structured_features(base)
        changed = base.copy()
        changed.loc[4, ["eps_surprise_raw", "revenue_surprise_raw",
                        "guidance_surprise_raw", "analyst_dispersion_raw"]] = -1_000.0
        second, second_stats = add_structured_features(changed)
        pd.testing.assert_series_equal(first.loc[:3, "eps_surprise_z"],
                                       second.loc[:3, "eps_surprise_z"])
        self.assertEqual(first_stats["features"]["eps_surprise_z"]["mean"],
                         second_stats["features"]["eps_surprise_z"]["mean"])

    def test_mismatch_is_fundamentals_minus_residual_reaction(self):
        frame = pd.DataFrame({
            "eps_surprise_z": [1.0, -1.0],
            "revenue_surprise_z": [0.5, -0.5],
            "guidance_surprise_z": [1.5, np.nan],
            "revision_breadth": [1.0, -0.5],
        })
        fundamental = calculate_fundamental_surprise(frame)
        mismatch = calculate_mismatch(fundamental, pd.Series([0.5, -0.25]))
        self.assertAlmostEqual(fundamental.iloc[0], 1.0)
        self.assertAlmostEqual(mismatch.iloc[0], 0.5)
        self.assertAlmostEqual(mismatch.iloc[1], -2 / 3 + 0.25)

    def test_m0_m1_m2_are_strictly_nested(self):
        frame = _coverage_frame()
        specs = _available_model_specs(
            frame[frame.time_bucket.eq("TRAIN_TIME")],
            frame[frame.time_bucket.eq("VALIDATION_TIME")],
        )
        self.assertEqual(list(specs), ["M0_price_reaction",
                                      "M1_structured_surprise",
                                      "M2_surprise_reaction_mismatch"])
        m0_num, m0_cat = specs["M0_price_reaction"]
        m1_num, m1_cat = specs["M1_structured_surprise"]
        m2_num, m2_cat = specs["M2_surprise_reaction_mismatch"]
        self.assertTrue(set(PRICE_NUMERIC).issubset(m0_num))
        self.assertTrue(set(m0_num).issubset(m1_num))
        self.assertTrue(set(m1_num).issubset(m2_num))
        self.assertTrue(set(PRICE_CATEGORICAL).issubset(m0_cat))
        self.assertTrue(set(m0_cat).issubset(m1_cat))
        self.assertTrue(set(m1_cat).issubset(m2_cat))

    def test_permutation_controls_break_event_alignment(self):
        class IdentityModel:
            @staticmethod
            def predict(features):
                return features["signal"].to_numpy()

        frame = pd.DataFrame({
            "signal": np.linspace(-1, 1, 40),
            "target_sector_residual_5d": np.linspace(-1, 1, 40),
        })
        result = permutation_controls(IdentityModel(), frame, ["signal"], [])
        self.assertEqual(result["status"], "COMPLETE")
        self.assertGreater(result["block_feature_permutation"]["rmse"], 0.1)
        self.assertGreater(result["shuffled_timestamp"]["rmse"], 0.1)

    def test_daily_ledger_marks_nav_and_includes_hedge_turnover(self):
        marks = json.dumps([
            {"date": "2025-01-02", "asset_close": 105.0,
             "benchmark_close": 100.0},
            {"date": "2025-01-03", "asset_close": 110.0,
             "benchmark_close": 100.0},
        ])
        trade = {
            "earnings_event_id": "event", "ticker": "TEST",
            "sector": "Technology", "entry_time": "2025-01-02T15:00:00Z",
            "exit_time": "2025-01-03T21:00:00Z", "side": 1,
            "weight": 0.05, "beta": 1.0, "entry_price": 100.0,
            "benchmark_entry_price": 100.0, "exit_price": 110.0,
            "benchmark_exit_price": 100.0, "daily_marks": marks,
        }
        result = PortfolioLedger(cost_bps_per_side=0).run([trade])
        self.assertEqual(len(result["daily_returns"]), 2)
        self.assertAlmostEqual(result["nav_path"][0]["nav"], 1.0025)
        self.assertAlmostEqual(result["net_return"], 0.005)
        self.assertGreater(result["hedge_turnover"], 0)
        self.assertGreater(result["max_gross_exposure"], 0.10)
        self.assertLess(result["max_gross_exposure"], 0.11)

    def test_signal_hurdle_is_twice_stock_and_hedge_cost(self):
        blocked = decision_from_prediction(0.011, beta=1.0)
        traded = decision_from_prediction(0.013, beta=1.0)
        self.assertEqual(blocked["side"], 0)
        self.assertEqual(traded["side"], 1)
        self.assertAlmostEqual(traded["hurdle_bps"], 120.0)


if __name__ == "__main__":
    unittest.main()
