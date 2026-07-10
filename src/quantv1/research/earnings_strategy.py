"""Shared earnings strategy decisions and bar-side execution rules.

The batch screen and localhost replay both call these functions.  This module
contains no model fitting and therefore cannot open the sealed holdout.
"""

from __future__ import annotations

import math


RESEARCH_MODEL_VERSION = "earnings-mismatch-elastic-net-v2"
RESEARCH_MODEL_LABEL = "RESEARCH MODEL — DATA GATE BLOCKED — NOT DEPLOYABLE"
BAR_COST_BPS_PER_SIDE = 15.0
COST_HURDLE_MULTIPLE = 2.0


def estimated_all_in_cost(beta: float | None,
                          cost_bps_per_side: float = BAR_COST_BPS_PER_SIDE) -> float | None:
    """Estimated stock-plus-hedge round-trip cost as a NAV-neutral return."""
    if beta is None or not math.isfinite(float(beta)):
        return None
    return 2.0 * float(cost_bps_per_side) / 1e4 * (1.0 + abs(float(beta)))


def decision_from_prediction(
        predicted_return: float | None, beta: float | None,
        cost_bps_per_side: float = BAR_COST_BPS_PER_SIDE,
        hurdle_multiple: float = COST_HURDLE_MULTIPLE,
        all_in_cost_estimate: float | None = None) -> dict:
    """Trade only when expected residual return clears the all-in cost hurdle."""
    if predicted_return is None or not math.isfinite(float(predicted_return)):
        return {"action": "NO TRADE", "side": 0,
                "reason": "prediction unavailable at this timestamp"}
    all_in_cost = (float(all_in_cost_estimate)
                   if all_in_cost_estimate is not None else
                   estimated_all_in_cost(beta, cost_bps_per_side))
    if all_in_cost is None:
        return {"action": "NO TRADE", "side": 0,
                "reason": "beta or hedge cost unavailable at this timestamp"}
    if hurdle_multiple < 1:
        raise ValueError("hurdle_multiple must be at least one")
    score_bps = float(predicted_return) * 1e4
    hurdle_bps = hurdle_multiple * all_in_cost * 1e4
    if abs(score_bps) <= hurdle_bps:
        return {"action": "NO TRADE", "side": 0,
                "reason": (f"score {score_bps:+.1f} bps is inside the "
                           f"±{hurdle_bps:.1f} bps cost hurdle"),
                "score_bps": score_bps, "hurdle_bps": hurdle_bps,
                "estimated_all_in_cost_bps": all_in_cost * 1e4}
    side = 1 if score_bps > 0 else -1
    return {"action": "BUY" if side > 0 else "SELL", "side": side,
            "reason": f"score {score_bps:+.1f} bps exceeds cost hurdle",
            "score_bps": score_bps, "hurdle_bps": hurdle_bps,
            "estimated_all_in_cost_bps": all_in_cost * 1e4}


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
    cost = estimated_all_in_cost(beta, cost_bps_per_side)
    if predicted_return is None or cost is None:
        return None
    return float(predicted_return) - cost
