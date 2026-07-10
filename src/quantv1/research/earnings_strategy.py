"""Shared earnings strategy decisions and bar-side execution rules.

The batch screen and localhost replay both call these functions.  This module
contains no model fitting and therefore cannot accidentally open the final test.
"""

from __future__ import annotations

import math


RESEARCH_MODEL_VERSION = "earnings-price-reaction-elastic-net-v1"
RESEARCH_MODEL_LABEL = "RESEARCH MODEL — REJECTED ON VALIDATION — NOT DEPLOYABLE"
SIGNAL_THRESHOLD_BPS = 10.0
BAR_COST_BPS_PER_SIDE = 15.0


def decision_from_prediction(predicted_return: float | None,
                              threshold_bps: float = SIGNAL_THRESHOLD_BPS) -> dict:
    """Map a model score to the canonical BUY/SELL/NO-TRADE decision."""
    if predicted_return is None or not math.isfinite(float(predicted_return)):
        return {"action": "NO TRADE", "side": 0,
                "reason": "prediction unavailable at this timestamp"}
    score_bps = float(predicted_return) * 1e4
    if abs(score_bps) <= threshold_bps:
        return {"action": "NO TRADE", "side": 0,
                "reason": f"score {score_bps:+.1f} bps is inside ±{threshold_bps:.1f} bps threshold",
                "score_bps": score_bps}
    side = 1 if score_bps > 0 else -1
    return {"action": "BUY" if side > 0 else "SELL", "side": side,
            "reason": f"score {score_bps:+.1f} bps exceeds threshold",
            "score_bps": score_bps}


def bar_entry_fill(open_price: float, side: int,
                   cost_bps_per_side: float = BAR_COST_BPS_PER_SIDE) -> float:
    """Conservative next-bar fill: pay adverse slippage in the trade direction."""
    return float(open_price) * (1.0 + int(side) * cost_bps_per_side / 1e4)


def bar_exit_fill(close_price: float, side: int,
                  cost_bps_per_side: float = BAR_COST_BPS_PER_SIDE) -> float:
    """Conservative exit fill: cross the market against the held position."""
    return float(close_price) * (1.0 - int(side) * cost_bps_per_side / 1e4)


def expected_after_cost(predicted_return: float | None, beta: float | None,
                        cost_bps_per_side: float = BAR_COST_BPS_PER_SIDE) -> float | None:
    """Display-only expected return after asset plus beta-scaled hedge costs."""
    if predicted_return is None:
        return None
    hedge = abs(float(beta)) if beta is not None else 0.0
    return float(predicted_return) - 2.0 * cost_bps_per_side / 1e4 * (1.0 + hedge)
