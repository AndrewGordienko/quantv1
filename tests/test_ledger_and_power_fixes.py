"""Regression tests for two committed protocol bugs.

1. PortfolioLedger must liquidate quote-complete trades at the trade's NBBO
   execution prices, not the exit-session bar close carried in ``daily_marks``.
2. The power gate must be coherent: effective sample size is executable trades
   deflated by the frozen design effect, with separate ticker/date minimums --
   not ``min(trades, tickers, dates)`` compared to the independent-sample count.
"""

from __future__ import annotations

import json
import unittest

from quantv1.portfolio.ledger import PortfolioLedger
from quantv1.research.protocol import (
    CLUSTER_DESIGN_EFFECT, evaluate_power, power_requirements,
)


class LedgerNbboExitTests(unittest.TestCase):
    """Quote-complete ledger P&L must reconcile with NBBO entry/exit + costs."""

    def _trade(self, *, exit_price: float, benchmark_exit_price: float,
               bar_close: float, benchmark_bar_close: float) -> dict:
        # daily_marks pre-populate the exit day with a *bar* close that differs
        # from the NBBO execution price. Pre-bug, setdefault kept the bar close.
        return {
            "earnings_event_id": "evt-1", "ticker": "TEST", "sector": "Technology",
            "entry_time": "2025-01-02T15:00:00Z", "exit_time": "2025-01-09T20:59:00Z",
            "announcement_date": "2025-01-02",
            "weight": 0.05, "side": 1, "beta": 1.0,
            "entry_price": 100.1, "benchmark_entry_price": 49.9,
            "exit_price": exit_price, "benchmark_exit_price": benchmark_exit_price,
            "ledger_cost_bps_per_side": 1.0,
            "daily_marks": json.dumps([
                {"date": "2025-01-09", "asset_close": bar_close,
                 "benchmark_close": benchmark_bar_close},
            ]),
        }

    @staticmethod
    def _expected_net_return(trade: dict, cost_rate: float) -> float:
        nav = 1.0
        weight, side, beta = trade["weight"], trade["side"], trade["beta"]
        asset_notional = nav * weight
        hedge_notional = asset_notional * abs(beta)
        hedge_side = -side if beta >= 0 else side
        asset_qty = side * asset_notional / trade["entry_price"]
        hedge_qty = hedge_side * hedge_notional / trade["benchmark_entry_price"]
        cash = nav
        cash -= asset_qty * trade["entry_price"]
        cash -= hedge_qty * trade["benchmark_entry_price"]
        cash -= cost_rate * (asset_notional + hedge_notional)          # entry cost
        cash += asset_qty * trade["exit_price"]
        cash += hedge_qty * trade["benchmark_exit_price"]
        asset_value = abs(asset_qty * trade["exit_price"])
        hedge_value = abs(hedge_qty * trade["benchmark_exit_price"])
        cash -= cost_rate * (asset_value + hedge_value)                # exit cost
        return cash / nav - 1.0

    def test_quote_complete_ledger_uses_nbbo_exit_not_bar_close(self):
        cost_rate = 1.0 / 1e4
        trade = self._trade(exit_price=105.0, benchmark_exit_price=51.0,
                            bar_close=110.0, benchmark_bar_close=53.0)
        result = PortfolioLedger().run([trade])
        expected = self._expected_net_return(trade, cost_rate)
        self.assertAlmostEqual(result["net_return"], expected, places=12)

        # And it must NOT equal the (wrong) bar-close liquidation.
        wrong = self._trade(exit_price=110.0, benchmark_exit_price=53.0,
                            bar_close=110.0, benchmark_bar_close=53.0)
        bar_close_return = self._expected_net_return(wrong, cost_rate)
        self.assertNotAlmostEqual(result["net_return"], bar_close_return, places=6)

        # Trade-level daily P&L must also sum to the ledger net return.
        pnl_sum = sum(row["pnl"] for row in result["trade_daily_pnl"])
        self.assertAlmostEqual(pnl_sum, result["net_return"], places=12)

    def test_bar_mode_exit_price_equals_bar_close_is_unchanged(self):
        # When exit_price already equals the bar close (BAR mode), forcing the
        # mark is a no-op: behaviour is identical to before.
        cost_rate = 1.0 / 1e4
        trade = self._trade(exit_price=110.0, benchmark_exit_price=53.0,
                            bar_close=110.0, benchmark_bar_close=53.0)
        result = PortfolioLedger().run([trade])
        self.assertAlmostEqual(result["net_return"],
                               self._expected_net_return(trade, cost_rate), places=12)


class CoherentPowerGateTests(unittest.TestCase):
    def test_power_gate_is_satisfiable_at_frozen_thresholds(self):
        power = power_requirements(0.04)
        self.assertEqual(power["status"], "FROZEN")
        # Frozen design at 4% vol: design effect 1.25, well above the naive
        # min-based rule that would demand ~independent_n unique tickers/dates.
        self.assertEqual(power["cluster_design_effect"], CLUSTER_DESIGN_EFFECT)
        trades = power["minimum_unique_executable_trades"]
        tickers = power["minimum_unique_tickers"]
        dates = power["minimum_announcement_dates"]
        # Sanity: the frozen minimums are far fewer tickers/dates than trades.
        self.assertLess(tickers, trades)
        self.assertLess(dates, trades)

        # A sample meeting exactly the frozen minimums must PASS -- the old
        # min(trades,tickers,dates) >= independent_n rule would have failed it.
        events_by_year = {"2024": power["minimum_events_per_eligible_year"]}
        gate = evaluate_power(power, unique_trades=trades, unique_tickers=tickers,
                              unique_dates=dates, events_by_year=events_by_year)
        self.assertTrue(gate["passes"], gate)
        # effective_n is trades / design_effect and clears independent_n.
        self.assertAlmostEqual(gate["effective_sample_size"],
                               trades / CLUSTER_DESIGN_EFFECT, places=9)
        self.assertGreaterEqual(gate["effective_sample_size"],
                                power["minimum_effective_sample_size"])

    def test_power_gate_fails_below_thresholds(self):
        power = power_requirements(0.04)
        events_by_year = {"2024": power["minimum_events_per_eligible_year"]}
        # Enough trades but too few tickers -> fails on the ticker minimum only.
        gate = evaluate_power(
            power, unique_trades=power["minimum_unique_executable_trades"],
            unique_tickers=power["minimum_unique_tickers"] - 1,
            unique_dates=power["minimum_announcement_dates"],
            events_by_year=events_by_year)
        self.assertFalse(gate["passes"])
        self.assertFalse(gate["checks"]["unique_tickers"])
        self.assertTrue(gate["checks"]["effective_sample_size"])

    def test_unavailable_power_never_passes(self):
        gate = evaluate_power({"status": "UNAVAILABLE"}, unique_trades=10_000,
                              unique_tickers=10_000, unique_dates=10_000,
                              events_by_year={"2024": 10_000})
        self.assertFalse(gate["passes"])


if __name__ == "__main__":
    unittest.main()
