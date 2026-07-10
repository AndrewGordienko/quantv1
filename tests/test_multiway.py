from __future__ import annotations

import unittest

import numpy as np

from quantv1.research.multiway import multiway_cluster_covariance


class MultiwayCovarianceTests(unittest.TestCase):
    def test_three_way_covariance_is_symmetric(self):
        rng = np.random.default_rng(4)
        x = np.column_stack([np.ones(120), rng.normal(size=(120, 2))])
        residuals = rng.normal(size=120)
        covariance = multiway_cluster_covariance(x, residuals, {
            "catalyst_day": np.repeat(np.arange(30), 4),
            "ticker": np.tile(np.arange(6), 20),
            "actor": np.tile(np.repeat(np.arange(5), 3), 8),
        })
        self.assertEqual(covariance.shape, (3, 3))
        np.testing.assert_allclose(covariance, covariance.T)
        self.assertTrue(np.isfinite(covariance).all())


if __name__ == "__main__":
    unittest.main()
