"""Tests for the NBBO data-contract skeleton (pure logic; no network/data)."""

import unittest

from quantv1.ingest import nbbo


class TestAggressor(unittest.TestCase):
    def test_buy_at_or_above_ask(self):
        self.assertEqual(nbbo.classify_aggressor(10.02, 10.00, 10.02), "BUY")
        self.assertEqual(nbbo.classify_aggressor(10.05, 10.00, 10.02), "BUY")

    def test_sell_at_or_below_bid(self):
        self.assertEqual(nbbo.classify_aggressor(10.00, 10.00, 10.02), "SELL")
        self.assertEqual(nbbo.classify_aggressor(9.98, 10.00, 10.02), "SELL")

    def test_above_midpoint_is_buy(self):
        # bid 10.00 ask 10.10 mid 10.05; 10.06 > mid -> BUY
        self.assertEqual(nbbo.classify_aggressor(10.06, 10.00, 10.10), "BUY")

    def test_midpoint_uses_tick_test(self):
        # exactly at midpoint -> Lee-Ready vs previous price
        self.assertEqual(nbbo.classify_aggressor(10.05, 10.00, 10.10, prev_price=10.04), "BUY")
        self.assertEqual(nbbo.classify_aggressor(10.05, 10.00, 10.10, prev_price=10.06), "SELL")
        self.assertIsNone(nbbo.classify_aggressor(10.05, 10.00, 10.10, prev_price=10.05))

    def test_crossed_or_locked_returns_none(self):
        self.assertIsNone(nbbo.classify_aggressor(10.00, 10.02, 10.00))   # crossed
        self.assertIsNone(nbbo.classify_aggressor(10.00, 10.00, 10.00))   # locked
        self.assertIsNone(nbbo.classify_aggressor(10.00, None, 10.02))


class TestStatus(unittest.TestCase):
    def test_status_touches_no_network(self):
        s = nbbo.status()
        self.assertEqual(s["tables"], ["quotes_nbbo", "trades_tick"])
        self.assertIn(s["state"], ("READY_TO_BACKFILL", "DATA_GATED_NO_KEY_OR_TIER"))

    def test_backfill_guarded_without_key(self):
        # never fabricates data; returns a guarded error state
        r = nbbo.backfill()
        self.assertIn("state", r)
        self.assertIn(r["state"], ("DATA_GATED", "READY_TO_IMPLEMENT"))


if __name__ == "__main__":
    unittest.main()
