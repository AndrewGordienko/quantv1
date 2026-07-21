"""Intraday fill simulator — pure, unit-testable, no market data.

Per docs/strategy_intraday.md this is built BEFORE any signal, because at intraday
horizons the fill IS the answer: a signal that is +18 bps under optimistic fills
and -6 bps under realistic fills is an assumption, not a discovery.

Design rules:
  * Marketable-limit only. You TAKE liquidity, crossing the spread — NO passive
    queue credit (you never earn the spread; assuming you do is the classic lie).
  * Explicit decision->exchange latency (default 50 ms); during it the quote can
    move against you (adverse selection), modeled as a bps drift of the touch.
  * Level-1 only: fill up to the displayed size at the touch; the remainder is
    unfilled (PARTIAL) unless an explicit depth-impact penalty is supplied.
  * Shorts require a locate; with none, the order is REJECTED (fail closed).
  * Costs are reported vs the midpoint so downstream code can enforce
    "net edge > 2x round-trip cost".

The SAME `simulate_fill` is called in historical replay and in live execution —
only the quote source (recorded vs live NBBO) changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Quote:
    """A point-in-time NBBO snapshot (level-1)."""
    bid: float
    bid_size: int
    ask: float
    ask_size: int
    ts: str | None = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def locked_or_crossed(self) -> bool:
        return not (self.ask > self.bid > 0)


@dataclass(frozen=True)
class Order:
    ticker: str
    side: str            # 'BUY' | 'SELL'
    qty: int
    limit_price: float | None = None   # None = marketable at the touch
    is_short: bool = False


@dataclass(frozen=True)
class Fill:
    filled_qty: int
    avg_price: float | None
    status: str          # 'FILLED' | 'PARTIAL' | 'REJECTED'
    cost_vs_mid_bps: float | None   # signed cost (positive = unfavorable vs mid)
    fee: float
    latency_ms: int
    reason: str


def simulate_fill(order: Order, quote: Quote, *, latency_ms: int = 50,
                  adverse_bps: float = 0.0, fee_bps: float = 0.2,
                  borrow_available: bool = True,
                  depth_impact_bps: float | None = None) -> Fill:
    """Simulate a marketable-limit fill against a single NBBO snapshot."""
    if quote.locked_or_crossed:
        return Fill(0, None, "REJECTED", None, 0.0, latency_ms, "locked_or_crossed_nbbo")
    if order.side not in ("BUY", "SELL"):
        return Fill(0, None, "REJECTED", None, 0.0, latency_ms, "bad_side")
    if order.side == "SELL" and order.is_short and not borrow_available:
        return Fill(0, None, "REJECTED", None, 0.0, latency_ms, "no_locate_fail_closed")
    if order.qty <= 0:
        return Fill(0, None, "REJECTED", None, 0.0, latency_ms, "non_positive_qty")

    buy = order.side == "BUY"
    touch = quote.ask if buy else quote.bid
    avail = quote.ask_size if buy else quote.bid_size
    # adverse drift of the touch during latency (price moves against the taker)
    eff_touch = touch * (1 + adverse_bps / 1e4) if buy else touch * (1 - adverse_bps / 1e4)

    # marketable-limit protection: reject if the (latency-adjusted) touch is worse
    # than the limit price
    if order.limit_price is not None:
        if buy and eff_touch > order.limit_price:
            return Fill(0, None, "REJECTED", None, 0.0, latency_ms, "limit_through_after_latency")
        if not buy and eff_touch < order.limit_price:
            return Fill(0, None, "REJECTED", None, 0.0, latency_ms, "limit_through_after_latency")

    at_touch = min(order.qty, max(0, avail))
    remainder = order.qty - at_touch
    legs = [(at_touch, eff_touch)] if at_touch > 0 else []

    if remainder > 0 and depth_impact_bps is not None:
        deeper = eff_touch * (1 + depth_impact_bps / 1e4) if buy else eff_touch * (1 - depth_impact_bps / 1e4)
        legs.append((remainder, deeper))
        remainder = 0

    filled = sum(q for q, _ in legs)
    if filled == 0:
        return Fill(0, None, "REJECTED", None, 0.0, latency_ms, "no_displayed_size")
    avg = sum(q * p for q, p in legs) / filled
    cost = ((avg / quote.mid) - 1) * 1e4 if buy else (1 - (avg / quote.mid)) * 1e4
    fee = filled * avg * fee_bps / 1e4
    status = "FILLED" if filled == order.qty else "PARTIAL"
    return Fill(filled, round(avg, 6), status, round(cost, 2), round(fee, 4),
                latency_ms, "ok")


def estimate_round_trip_cost_bps(quote: Quote, *, fee_bps: float = 0.2,
                                 adverse_bps: float = 0.0) -> float:
    """Rough entry+exit cost (bps of notional) for the net-edge > 2x cost gate:
    two spread crossings (half-spread each way vs mid) + two fees + two adverse drifts."""
    half_spread_bps = (quote.ask - quote.mid) / quote.mid * 1e4
    return round(2 * (half_spread_bps + fee_bps + adverse_bps), 2)
