"""Tests for the intraday fill simulator (pure logic; no market data)."""

import unittest

from quantv1.execution.fill_sim import (Quote, Order, simulate_fill,
                                        estimate_round_trip_cost_bps)


Q = Quote(bid=10.00, bid_size=500, ask=10.02, ask_size=500)   # mid 10.01


class TestFillSim(unittest.TestCase):
    def test_marketable_buy_crosses_spread_no_queue_credit(self):
        f = simulate_fill(Order("X", "BUY", 300), Q, fee_bps=0.0, adverse_bps=0.0)
        self.assertEqual(f.status, "FILLED")
        self.assertEqual(f.avg_price, 10.02)          # pays the ask, not the mid
        self.assertGreater(f.cost_vs_mid_bps, 0)      # crossing costs vs mid

    def test_partial_when_qty_exceeds_displayed_size(self):
        f = simulate_fill(Order("X", "BUY", 800), Q)
        self.assertEqual(f.status, "PARTIAL")
        self.assertEqual(f.filled_qty, 500)           # only top-of-book size

    def test_depth_impact_fills_remainder_worse(self):
        f = simulate_fill(Order("X", "BUY", 800), Q, depth_impact_bps=10.0, fee_bps=0.0)
        self.assertEqual(f.status, "FILLED")
        self.assertGreater(f.avg_price, 10.02)        # blended above the ask

    def test_latency_adverse_selection_raises_cost(self):
        base = simulate_fill(Order("X", "BUY", 100), Q, adverse_bps=0.0, fee_bps=0.0)
        adv = simulate_fill(Order("X", "BUY", 100), Q, adverse_bps=5.0, fee_bps=0.0)
        self.assertGreater(adv.avg_price, base.avg_price)

    def test_limit_protection_rejects_when_touch_moves_through(self):
        # marketable buy limit at the ask; a 20 bps adverse drift pushes past it
        f = simulate_fill(Order("X", "BUY", 100, limit_price=10.02), Q, adverse_bps=20.0)
        self.assertEqual(f.status, "REJECTED")
        self.assertEqual(f.reason, "limit_through_after_latency")

    def test_short_without_locate_fails_closed(self):
        f = simulate_fill(Order("X", "SELL", 100, is_short=True), Q, borrow_available=False)
        self.assertEqual(f.status, "REJECTED")
        self.assertEqual(f.reason, "no_locate_fail_closed")

    def test_locked_or_crossed_rejected(self):
        crossed = Quote(bid=10.02, bid_size=100, ask=10.00, ask_size=100)
        self.assertEqual(simulate_fill(Order("X", "BUY", 100), crossed).status, "REJECTED")

    def test_sell_hits_bid(self):
        f = simulate_fill(Order("X", "SELL", 100), Q, fee_bps=0.0)
        self.assertEqual(f.avg_price, 10.00)          # sells at the bid
        self.assertGreater(f.cost_vs_mid_bps, 0)

    def test_round_trip_cost_is_positive(self):
        self.assertGreater(estimate_round_trip_cost_bps(Q, fee_bps=0.2, adverse_bps=1.0), 0)


if __name__ == "__main__":
    unittest.main()
