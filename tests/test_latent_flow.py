from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quantv1.research.latent_flow import _episode_starts, _regular_session, _two_factor_residual


class LatentFlowTests(unittest.TestCase):
    def test_regular_session_filter_is_dst_safe(self):
        # July is EDT: 13:30 UTC is the cash open, 20:00 UTC is the close.
        bars = pd.DataFrame({
            "ticker": ["X"] * 4,
            "ts": pd.to_datetime(["2026-07-06 13:29:00", "2026-07-06 13:30:00",
                                  "2026-07-06 19:59:00", "2026-07-06 20:00:00"]),
        })
        got = _regular_session(bars)
        self.assertEqual(got["slot"].tolist(), [0, 389])

    def test_factor_betas_do_not_use_current_return(self):
        idx = pd.date_range("2026-01-02 14:30", periods=8, freq="min")
        market = pd.Series([.001, -.001, .002, .001, -.002, .001, .002, -.001], index=idx)
        sector = pd.DataFrame({"X": [.0005, .001, -.001, .0002, .0005, -.0004, .0001, .0003]}, index=idx)
        y = pd.DataFrame({"X": 1.2 * market + .7 * sector["X"]}, index=idx)
        _, beta_a, _ = _two_factor_residual(y, market, sector, window=3)
        changed = y.copy()
        changed.iloc[5, 0] += .25
        _, beta_b, _ = _two_factor_residual(changed, market, sector, window=3)
        # The beta used at t=5 only has observations through t=4.
        self.assertAlmostEqual(beta_a.iloc[5, 0], beta_b.iloc[5, 0], places=12)

    def test_episode_dedup_keeps_first_signal_after_cooldown(self):
        base = pd.Timestamp("2026-01-02 14:30")
        candidates = pd.DataFrame({
            "ticker": ["A", "A", "A", "B"],
            "ts": [base, base + pd.Timedelta(minutes=30), base + pd.Timedelta(minutes=60), base],
        })
        got = _episode_starts(candidates, cooldown_minutes=60)
        self.assertEqual(list(got["ticker"]), ["A", "A", "B"])
        self.assertEqual(list(got.loc[got["ticker"] == "A", "ts"]), [base, base + pd.Timedelta(minutes=60)])


if __name__ == "__main__":
    unittest.main()
