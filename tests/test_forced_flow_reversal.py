"""Tests for the Track A reversal harness (research.forced_flow_reversal)."""

from __future__ import annotations

import unittest

import numpy as np

from quantv1.research import forced_flow_reversal as fr


def _panel(stock_close, spy_close):
    dates = np.array([np.datetime64("2024-01-01") + np.timedelta64(i, "D")
                      for i in range(len(spy_close))])
    def bars(close):
        close = np.asarray(close, dtype=float)
        return {"date": dates, "open": close.copy(), "close": close.copy(),
                "dollar": np.ones(len(close)) * 1e6}
    return {"TEST": bars(stock_close), fr.BENCHMARK_TICKER: bars(spy_close)}


class ResidualTests(unittest.TestCase):
    def test_window_return_open_to_close(self):
        o = np.array([10.0, 11.0]); c = np.array([10.5, 12.0])
        self.assertAlmostEqual(fr._window_return(o, c, 0, 1), 12.0 / 10.0 - 1.0)

    def test_identical_series_has_zero_residual(self):
        rng = np.random.default_rng(1)
        spy = 100 * np.cumprod(1 + rng.normal(0, 0.01, 120))
        panel = _panel(spy, spy)                     # stock == market -> beta 1, residual 0
        beta = fr._beta(panel, "TEST", 80)
        self.assertAlmostEqual(beta, 1.0, places=6)
        res = fr._residual(panel, "TEST", 80, beta, 1, 5)
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res, 0.0, places=9)

    def test_idiosyncratic_move_shows_in_residual(self):
        rng = np.random.default_rng(2)
        spy = 100 * np.cumprod(1 + rng.normal(0, 0.005, 120))
        stock = spy.copy()
        # +3% idiosyncratic bump on the D+5 exit bar only
        stock[86:] = stock[86:] * 1.03
        panel = _panel(stock, spy)
        beta = fr._beta(panel, "TEST", 80)
        res = fr._residual(panel, "TEST", 80, beta, 1, 5)   # exit idx 85 -> before bump
        res_bump = fr._residual(panel, "TEST", 80, beta, 1, 6)  # exit idx 86 -> includes bump
        self.assertLess(res, 0.02)
        self.assertGreater(res_bump, 0.02)


class ClusterBootstrapTests(unittest.TestCase):
    def test_mean_and_batch_count(self):
        rng = np.random.default_rng(3)
        values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        batches = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"]
        out = fr._cluster_bootstrap(values, batches, rng)
        self.assertEqual(out["n_batches"], 6)
        self.assertEqual(out["n_obs"], 12)
        self.assertAlmostEqual(out["mean_bps"], np.mean(values) * 1e4, places=3)
        self.assertLessEqual(out["ci_low"], out["mean_bps"])
        self.assertGreaterEqual(out["ci_high"], out["mean_bps"])

    def test_ignores_none_and_reports_shortfall(self):
        rng = np.random.default_rng(4)
        out = fr._cluster_bootstrap([0.01, None, 0.02], ["a", "b", "c"], rng)
        self.assertEqual(out["n_obs"], 2)
        self.assertIsNone(out["ci_low"])           # too few for a bootstrap CI


if __name__ == "__main__":
    unittest.main()
