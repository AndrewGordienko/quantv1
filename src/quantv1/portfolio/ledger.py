"""Daily mark-to-market accounting for event-driven stock/hedge pairs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import math
from typing import Iterable

import pandas as pd


def _as_date(value) -> date:
    return pd.Timestamp(value).date()


def _marks(value) -> dict[date, tuple[float, float]]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    records = json.loads(value) if isinstance(value, str) else value
    result = {}
    for record in records or []:
        asset = float(record["asset_close"])
        benchmark = float(record["benchmark_close"])
        if asset > 0 and benchmark > 0:
            result[_as_date(record["date"])] = (asset, benchmark)
    return result


@dataclass
class _Position:
    trade: dict
    asset_quantity: float
    hedge_quantity: float
    marks: dict[date, tuple[float, float]]
    last_asset: float
    last_benchmark: float


class PortfolioLedger:
    """Account for a set of pre-approved stock/sector-hedge trades.

    The ledger uses fixed quantities established from NAV at entry. Costs and
    turnover include both the stock and beta hedge. Exposure is observed at
    each session close before scheduled close orders are applied, so an exit
    day is not incorrectly reported as a zero-risk day.
    """

    def __init__(self, *, initial_nav: float = 1.0,
                 cost_bps_per_side: float = 15.0,
                 calendar: Iterable[date] | None = None):
        if initial_nav <= 0:
            raise ValueError("initial_nav must be positive")
        if cost_bps_per_side < 0:
            raise ValueError("cost_bps_per_side cannot be negative")
        self.initial_nav = float(initial_nav)
        self.cost_rate = float(cost_bps_per_side) / 1e4
        self.calendar = {_as_date(day) for day in (calendar or [])}

    @staticmethod
    def _position_marks(trade: dict) -> dict[date, tuple[float, float]]:
        marks = _marks(trade.get("daily_marks"))
        exit_day = _as_date(trade["exit_time"])
        marks.setdefault(exit_day, (float(trade["exit_price"]),
                                    float(trade["benchmark_exit_price"])))
        return marks

    @staticmethod
    def _exposure(positions: list[_Position], nav: float) -> dict:
        if nav <= 0:
            return {"gross": math.inf, "net": math.inf,
                    "sector_gross": {}}
        gross = net = 0.0
        sectors: dict[str, float] = {}
        for position in positions:
            asset_value = position.asset_quantity * position.last_asset
            hedge_value = position.hedge_quantity * position.last_benchmark
            gross += abs(asset_value) + abs(hedge_value)
            net += asset_value + hedge_value
            sector = str(position.trade["sector"])
            sectors[sector] = sectors.get(sector, 0.0) + abs(asset_value) / nav
        return {"gross": gross / nav, "net": net / nav,
                "sector_gross": sectors}

    def run(self, trades: list[dict]) -> dict:
        if not trades:
            return {
                "n_trades": 0, "net_return": 0.0, "daily_returns": [],
                "nav_path": [], "exposure_path": [], "max_drawdown": 0.0,
                "turnover": 0.0, "stock_turnover": 0.0,
                "hedge_turnover": 0.0, "avg_gross_exposure": 0.0,
                "max_gross_exposure": 0.0, "max_net_exposure": 0.0,
                "max_sector_gross_exposure": 0.0, "trades": [],
            }

        by_entry: dict[date, list[dict]] = {}
        days = set(self.calendar)
        for trade in trades:
            entry_day = _as_date(trade["entry_time"])
            by_entry.setdefault(entry_day, []).append(trade)
            days.add(entry_day)
            days.add(_as_date(trade["exit_time"]))
            days.update(self._position_marks(trade))
        first_day = min(by_entry)
        last_day = max(_as_date(trade["exit_time"]) for trade in trades)
        days = sorted(day for day in days if first_day <= day <= last_day)

        cash = self.initial_nav
        previous_nav = self.initial_nav
        active: list[_Position] = []
        nav_path = []
        exposure_path = []
        daily_returns = []
        total_stock_turnover = total_hedge_turnover = 0.0

        for day in days:
            day_stock_turnover = day_hedge_turnover = 0.0
            for trade in by_entry.get(day, []):
                weight = float(trade["weight"])
                side = int(trade["side"])
                beta = float(trade["beta"])
                asset_entry = float(trade["entry_price"])
                benchmark_entry = float(trade["benchmark_entry_price"])
                asset_notional = previous_nav * weight
                hedge_notional = asset_notional * abs(beta)
                hedge_side = -side if beta >= 0 else side
                asset_quantity = side * asset_notional / asset_entry
                hedge_quantity = (hedge_side * hedge_notional /
                                  benchmark_entry if hedge_notional else 0.0)
                cash -= asset_quantity * asset_entry
                cash -= hedge_quantity * benchmark_entry
                cash -= self.cost_rate * (asset_notional + hedge_notional)
                day_stock_turnover += asset_notional
                day_hedge_turnover += hedge_notional
                active.append(_Position(
                    trade=trade, asset_quantity=asset_quantity,
                    hedge_quantity=hedge_quantity,
                    marks=self._position_marks(trade),
                    last_asset=asset_entry, last_benchmark=benchmark_entry,
                ))

            for position in active:
                mark = position.marks.get(day)
                if mark:
                    position.last_asset, position.last_benchmark = mark

            pre_exit_nav = cash + sum(
                position.asset_quantity * position.last_asset +
                position.hedge_quantity * position.last_benchmark
                for position in active
            )
            exposure = self._exposure(active, pre_exit_nav)
            closing = [position for position in active
                       if _as_date(position.trade["exit_time"]) == day]
            for position in closing:
                asset_value = abs(position.asset_quantity * position.last_asset)
                hedge_value = abs(position.hedge_quantity * position.last_benchmark)
                cash += position.asset_quantity * position.last_asset
                cash += position.hedge_quantity * position.last_benchmark
                cash -= self.cost_rate * (asset_value + hedge_value)
                day_stock_turnover += asset_value
                day_hedge_turnover += hedge_value
            if closing:
                closing_ids = {id(position) for position in closing}
                active = [position for position in active
                          if id(position) not in closing_ids]

            nav = cash + sum(
                position.asset_quantity * position.last_asset +
                position.hedge_quantity * position.last_benchmark
                for position in active
            )
            daily_return = nav / previous_nav - 1.0
            stock_turnover = day_stock_turnover / previous_nav
            hedge_turnover = day_hedge_turnover / previous_nav
            nav_path.append({"date": str(day), "nav": float(nav),
                             "return": float(daily_return)})
            exposure_path.append({
                "date": str(day), "gross": float(exposure["gross"]),
                "net": float(exposure["net"]),
                "sector_gross": exposure["sector_gross"],
                "stock_turnover": float(stock_turnover),
                "hedge_turnover": float(hedge_turnover),
            })
            daily_returns.append(float(daily_return))
            total_stock_turnover += stock_turnover
            total_hedge_turnover += hedge_turnover
            previous_nav = nav

        nav_series = pd.Series(
            [self.initial_nav] + [row["nav"] for row in nav_path], dtype=float
        )
        drawdown = nav_series / nav_series.cummax() - 1.0
        gross_path = [row["gross"] for row in exposure_path]
        net_path = [abs(row["net"]) for row in exposure_path]
        sector_path = [value for row in exposure_path
                       for value in row["sector_gross"].values()]
        return {
            "n_trades": len(trades),
            "net_return": float(previous_nav / self.initial_nav - 1.0),
            "daily_returns": daily_returns, "nav_path": nav_path,
            "exposure_path": exposure_path,
            "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
            "turnover": float(total_stock_turnover + total_hedge_turnover),
            "stock_turnover": float(total_stock_turnover),
            "hedge_turnover": float(total_hedge_turnover),
            "avg_gross_exposure": float(pd.Series(gross_path).mean()),
            "max_gross_exposure": float(max(gross_path, default=0.0)),
            "max_net_exposure": float(max(net_path, default=0.0)),
            "max_sector_gross_exposure": float(max(sector_path, default=0.0)),
            "trades": trades,
        }
