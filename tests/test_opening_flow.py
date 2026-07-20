from __future__ import annotations

import unittest

import pandas as pd

from quantv1.forward.opening_flow import POLICY, _select, version
from quantv1.forward.opening_flow_live import _snapshot_quotes


class OpeningFlowTests(unittest.TestCase):
    def test_policy_version_is_frozen_and_cost_gate_is_explicit(self):
        self.assertTrue(version().startswith("opening-flow-v1-"))
        self.assertEqual(POLICY["max_positions"], 1)
        row = pd.Series({"gap_z": 1.2, "gap": 0.01, "residual": 0.004,
                         "residual_z": 0.4, "peer_z": 0.5, "peer_return": 0.003,
                         "relative_volume": 1.4, "liquidity_ok": True})
        self.assertTrue(_select(row, "P1"))
        self.assertFalse(_select(row, "P2"))  # residual z does not meet the frozen gate

    def test_quote_gate_calculates_spread(self):
        q = _snapshot_quotes({"A": {"latestQuote": {"bp": 100, "ap": 100.10,
                                                        "bs": 10, "as": 12}}})["A"]
        self.assertAlmostEqual(q["mid"], 100.05)
        self.assertGreater(q["spread_bps"], 9)


if __name__ == "__main__":
    unittest.main()
