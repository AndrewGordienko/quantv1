from __future__ import annotations

import unittest

import numpy as np

from quantv1.v4.actor_impact import _DAY_NS, _MINUTE_NS, _outcome
from quantv1.v4.replay import BarPanel


def _panel(days=1, minutes=240):
    panel = object.__new__(BarPanel)
    timestamps = []
    for day in range(days):
        base = (20_000 + day) * _DAY_NS + 14 * 60 * _MINUTE_NS
        timestamps.extend(base + np.arange(minutes) * _MINUTE_NS)
    ts = np.asarray(timestamps, dtype=np.int64)
    pattern = np.sin(np.arange(len(ts) - 1) / 7) * 0.00005
    market = 100 * np.cumprod(np.r_[1.0, 1.0001 + pattern])
    stock = 50 * np.cumprod(np.r_[1.0, 1.0002 + 1.5 * pattern])
    sector = 75 * np.cumprod(np.r_[1.0, 1.00015 + 1.2 * pattern])

    def bars(close):
        return {"ts": ts, "open": close.copy(), "high": close * 1.001,
                "low": close * 0.999, "close": close.copy(),
                "vol": np.ones(len(ts))}
    panel.data = {"TEST": bars(stock), "SPY": bars(market), "XLK": bars(sector)}
    return panel


class ActorOutcomeTests(unittest.TestCase):
    def test_rejects_horizon_that_crosses_session(self):
        panel = _panel(days=2, minutes=240)
        event_time = int(panel.data["TEST"]["ts"][230] - 1)
        self.assertIsNone(_outcome(panel, "TEST", event_time, 30, "XLK"))

    def test_reports_raw_residual_and_nonexecuted_hedge_separately(self):
        panel = _panel(days=1, minutes=240)
        event_time = int(panel.data["TEST"]["ts"][150] - 1)
        outcome = _outcome(panel, "TEST", event_time, 30, "XLK")
        self.assertIsNotNone(outcome)
        self.assertTrue(outcome["same_session"])
        self.assertIsNotNone(outcome["raw_return"])
        self.assertIsNotNone(outcome["market_beta_residual"])
        self.assertIsNotNone(outcome["sector_beta_residual"])
        self.assertIsNone(outcome["actually_hedged_return"])
        self.assertEqual(outcome["hedge_execution_status"], "NOT_EXECUTED")
        self.assertFalse(outcome["executable"])


if __name__ == "__main__":
    unittest.main()
