"""Frozen statistical, calendar, beta, and execution protocol for EERM."""

from __future__ import annotations

from functools import lru_cache
import math

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from scipy.stats import norm


TARGET_VERSION = "beta-hedged-mid-v1"
BETA_VERSION = "pre-event-60d-shrunk-clipped-v1"
BETA_PRIOR = 1.0
BETA_PRIOR_OBSERVATIONS = 20
BETA_MIN = 0.0
BETA_MAX = 2.0
HAC_LAG = 5
BOOTSTRAP_DRAWS = 1_000
MIN_PERMUTATION_FRACTION = 0.50
MIN_ECONOMIC_NET_RETURN = 0.01
POWER_ALPHA = 0.05
POWER_LEVEL = 0.80
CLUSTER_DESIGN_EFFECT = 1.25
MAX_EVENTS_PER_TICKER = 4
MAX_EVENTS_PER_ANNOUNCEMENT_DATE = 8
MIN_EVENTS_PER_ELIGIBLE_YEAR = 30
RESEARCH_NAV_USD = 1_000_000.0
POSITION_WEIGHT = 0.05
ADVERSE_SELECTION_BPS_PER_SIDE = 1.0
IMPACT_COEFFICIENT_BPS = 20.0
BAR_COST_BPS_PER_SIDE = 15.0
HOLDING_SESSIONS = 5


@lru_cache(maxsize=1)
def xnys_calendar():
    return xcals.get_calendar("XNYS")


def trading_sessions(start, end) -> list:
    start_day = pd.Timestamp(start).date()
    end_day = pd.Timestamp(end).date()
    sessions = xnys_calendar().sessions_in_range(str(start_day), str(end_day))
    return [pd.Timestamp(session).date() for session in sessions]


def first_session_after(timestamp) -> str:
    day = pd.Timestamp(timestamp).date()
    sessions = trading_sessions(day + pd.Timedelta(days=1),
                                day + pd.Timedelta(days=14))
    if not sessions:
        raise ValueError("no XNYS session found after model lock")
    return str(sessions[0])


def shrink_and_clip_beta(raw_beta: float, observations: int) -> float:
    if not np.isfinite(raw_beta) or observations < 1:
        return BETA_PRIOR
    reliability = observations / (observations + BETA_PRIOR_OBSERVATIONS)
    shrunk = reliability * float(raw_beta) + (1.0 - reliability) * BETA_PRIOR
    return float(np.clip(shrunk, BETA_MIN, BETA_MAX))


def hac_statistics(daily_returns: list[float], *, lag: int = HAC_LAG) -> dict:
    values = np.asarray(daily_returns, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n_sessions": 0, "lag": lag, "mean_daily": None,
                "annualized_alpha": None, "annualized_alpha_ci": [None, None],
                "mean_t_stat": None, "sharpe_annual": None}
    mean = float(values.mean())
    centered = values - mean
    n = len(values)
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    for offset in range(1, min(lag, n - 1) + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / n)
        weight = 1.0 - offset / (lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    variance_of_mean = max(long_run_variance / n, 0.0)
    standard_error = math.sqrt(variance_of_mean)
    z = norm.ppf(0.975)
    lower = (mean - z * standard_error) * 252
    upper = (mean + z * standard_error) * 252
    sample_std = values.std(ddof=1) if n > 1 else 0.0
    return {
        "n_sessions": n, "lag": lag, "mean_daily": mean,
        "mean_standard_error": standard_error,
        "mean_t_stat": mean / standard_error if standard_error > 0 else None,
        "annualized_alpha": mean * 252,
        "annualized_alpha_ci": [float(lower), float(upper)],
        "sharpe_annual": (float(mean / sample_std * np.sqrt(252))
                          if sample_std > 0 else None),
    }


def power_requirements(target_volatility: float, *,
                       effect: float = MIN_ECONOMIC_NET_RETURN) -> dict:
    volatility = float(target_volatility)
    if not np.isfinite(volatility) or volatility <= 0 or effect <= 0:
        return {"status": "UNAVAILABLE"}
    independent_n = math.ceil((
        (norm.ppf(1 - POWER_ALPHA / 2) + norm.ppf(POWER_LEVEL)) *
        volatility / effect
    ) ** 2)
    executable_trades = math.ceil(independent_n * CLUSTER_DESIGN_EFFECT)
    return {
        "status": "FROZEN", "alpha": POWER_ALPHA, "power": POWER_LEVEL,
        "minimum_economic_net_return": effect,
        "target_volatility": volatility,
        "cluster_design_effect": CLUSTER_DESIGN_EFFECT,
        "minimum_unique_executable_trades": executable_trades,
        "minimum_unique_tickers": math.ceil(
            executable_trades / MAX_EVENTS_PER_TICKER
        ),
        "minimum_announcement_dates": math.ceil(
            executable_trades / MAX_EVENTS_PER_ANNOUNCEMENT_DATE
        ),
        "minimum_effective_sample_size": independent_n,
        "minimum_events_per_eligible_year": MIN_EVENTS_PER_ELIGIBLE_YEAR,
    }


def evaluate_power(power: dict, *, unique_trades: int, unique_tickers: int,
                   unique_dates: int, events_by_year: dict) -> dict:
    """Coherent power gate for the frozen design.

    ``power_requirements`` already permits up to ``MAX_EVENTS_PER_TICKER`` and
    ``MAX_EVENTS_PER_ANNOUNCEMENT_DATE`` events per cluster and inflates the
    independent sample by ``CLUSTER_DESIGN_EFFECT``. The effective (independent)
    sample size is therefore the executable-trade count deflated by that design
    effect -- NOT ``min(trades, tickers, dates)``, which would silently demand
    one ticker and one date per independent observation and contradict the
    frozen ticker/date minimums. The clustered bootstrap CI is a separate,
    complementary check reported alongside this gate.
    """
    if power.get("status") != "FROZEN":
        return {"effective_sample_size": 0.0, "passes": False,
                "reason": "power requirements unavailable"}
    design_effect = float(power.get("cluster_design_effect", CLUSTER_DESIGN_EFFECT))
    effective_n = unique_trades / design_effect if design_effect > 0 else 0.0
    checks = {
        "unique_executable_trades": unique_trades >=
            power["minimum_unique_executable_trades"],
        "unique_tickers": unique_tickers >= power["minimum_unique_tickers"],
        "announcement_dates": unique_dates >= power["minimum_announcement_dates"],
        "effective_sample_size": effective_n >=
            power["minimum_effective_sample_size"],
        "events_per_eligible_year": bool(events_by_year) and
            min(events_by_year.values()) >= power["minimum_events_per_eligible_year"],
    }
    return {"effective_sample_size": float(effective_n),
            "checks": checks, "passes": all(checks.values())}


def _cluster_weights(frame: pd.DataFrame, rng) -> np.ndarray:
    dates = frame["announcement_date"].astype(str)
    tickers = frame["ticker"].astype(str)
    unique_dates = dates.unique()
    unique_tickers = tickers.unique()
    sampled_dates = rng.choice(unique_dates, len(unique_dates), replace=True)
    sampled_tickers = rng.choice(unique_tickers, len(unique_tickers), replace=True)
    date_counts = pd.Series(sampled_dates).value_counts()
    ticker_counts = pd.Series(sampled_tickers).value_counts()
    return np.asarray([
        float(date_counts.get(day, 0) * ticker_counts.get(ticker, 0))
        for day, ticker in zip(dates, tickers)
    ])


def clustered_mean_ci(frame: pd.DataFrame, values, *,
                      draws: int = BOOTSTRAP_DRAWS,
                      random_state: int = 23) -> dict:
    data = frame.copy().reset_index(drop=True)
    data["value"] = np.asarray(values, dtype=float)
    data = data[np.isfinite(data["value"])].reset_index(drop=True)
    if "announcement_date" not in data:
        data["announcement_date"] = pd.to_datetime(
            data["entry_time"], utc=True
        ).dt.date.astype(str)
    if len(data) < 2 or data.ticker.nunique() < 2 or data.announcement_date.nunique() < 2:
        return {"status": "INSUFFICIENT_CLUSTERS", "estimate": None,
                "confidence_interval": [None, None],
                "effective_sample_size": 0}
    rng = np.random.default_rng(random_state)
    estimates = []
    for _ in range(draws):
        weights = _cluster_weights(data, rng)
        if weights.sum() > 0:
            estimates.append(float(np.average(data["value"], weights=weights)))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    effective = min(len(data), data.ticker.nunique(),
                    data.announcement_date.nunique())
    return {
        "status": "COMPLETE", "draws": len(estimates),
        "clusters": {"ticker": int(data.ticker.nunique()),
                     "announcement_date": int(data.announcement_date.nunique())},
        "estimate": float(data["value"].mean()),
        "confidence_interval": [float(lower), float(upper)],
        "effective_sample_size": int(effective),
    }


def clustered_portfolio_bootstrap(portfolio: dict, *,
                                  draws: int = BOOTSTRAP_DRAWS,
                                  random_state: int = 23) -> dict:
    """Two-way pigeonhole bootstrap over announcement sessions and tickers."""
    trades = pd.DataFrame(portfolio.get("trades", []))
    attribution = pd.DataFrame(portfolio.get("trade_daily_pnl", []))
    if (len(trades) < 2 or attribution.empty or trades.ticker.nunique() < 2):
        return {"status": "INSUFFICIENT_CLUSTERS",
                "confidence_intervals": {}, "effective_sample_size": 0}
    if "announcement_date" not in trades:
        trades["announcement_date"] = pd.to_datetime(
            trades["entry_time"], utc=True
        ).dt.date.astype(str)
    trades["earnings_event_id"] = trades["earnings_event_id"].astype(str)
    attribution["earnings_event_id"] = attribution["earnings_event_id"].astype(str)
    calendar = [row["date"] for row in portfolio.get("nav_path", [])]
    if not calendar:
        return {"status": "NO_CALENDAR", "confidence_intervals": {},
                "effective_sample_size": 0}
    matrix = attribution.pivot_table(
        index="earnings_event_id", columns="date", values="pnl",
        aggfunc="sum", fill_value=0.0,
    ).reindex(index=trades["earnings_event_id"], columns=calendar,
              fill_value=0.0).to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    samples = {name: [] for name in (
        "mean_net_return_per_trade", "total_portfolio_return",
        "annualized_alpha", "sharpe_annual",
    )}
    for _ in range(draws):
        weights = _cluster_weights(trades, rng)
        if weights.sum() <= 0:
            continue
        daily_pnl = weights @ matrix
        nav = 1.0
        returns = []
        for pnl in daily_pnl:
            returns.append(float(pnl / nav))
            nav += float(pnl)
        values = np.asarray(returns)
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        samples["mean_net_return_per_trade"].append(
            float(daily_pnl.sum() / weights.sum())
        )
        samples["total_portfolio_return"].append(float(nav - 1.0))
        samples["annualized_alpha"].append(float(values.mean() * 252))
        samples["sharpe_annual"].append(
            float(values.mean() / std * np.sqrt(252)) if std > 0 else 0.0
        )
    intervals = {
        name: [float(value) for value in np.quantile(series, [0.025, 0.975])]
        for name, series in samples.items() if series
    }
    return {
        "status": "COMPLETE", "draws": len(next(iter(samples.values()))),
        "clusters": {"ticker": int(trades.ticker.nunique()),
                     "announcement_date": int(trades.announcement_date.nunique())},
        "confidence_intervals": intervals,
        "effective_sample_size": int(min(
            len(trades), trades.ticker.nunique(),
            trades.announcement_date.nunique(),
        )),
    }


def _half_spread_bps(bid, ask) -> float | None:
    if not all(value is not None and np.isfinite(value) and value > 0
               for value in (bid, ask)) or ask < bid:
        return None
    mid = (float(bid) + float(ask)) / 2.0
    return (float(ask) - float(bid)) / (2.0 * mid) * 1e4


def execution_cost_estimate(row: dict, side: int, *, delayed: bool = False,
                            doubled: bool = False) -> dict:
    try:
        beta = abs(float(row["beta"]))
    except (KeyError, TypeError, ValueError):
        return {"deployable": False, "reason": "frozen beta unavailable"}
    if not np.isfinite(beta):
        return {"deployable": False, "reason": "frozen beta unavailable"}
    borrow_available = row.get("borrow_available")
    if (side < 0 and
            (borrow_available is None or pd.isna(borrow_available) or
             not bool(borrow_available))):
        return {"deployable": False, "reason": "historical stock borrow unavailable"}
    borrow_fee = float(row.get("borrow_fee_bps_annual") or 0.0)
    if side < 0 and (not np.isfinite(borrow_fee) or borrow_fee < 0):
        return {"deployable": False, "reason": "historical stock borrow fee invalid"}

    entry_prefix = "delayed_entry" if delayed else "entry"
    spreads = [
        _half_spread_bps(row.get(f"{entry_prefix}_bid"),
                         row.get(f"{entry_prefix}_ask")),
        _half_spread_bps(row.get("exit_bid"), row.get("exit_ask")),
        _half_spread_bps(row.get(f"benchmark_{entry_prefix}_bid"),
                         row.get(f"benchmark_{entry_prefix}_ask")),
        _half_spread_bps(row.get("benchmark_exit_bid"),
                         row.get("benchmark_exit_ask")),
    ]
    quote_mode = bool(row.get("quote_complete")) and all(
        value is not None for value in spreads
    )
    if quote_mode:
        embedded_spread = (spreads[0] + spreads[1] +
                           beta * (spreads[2] + spreads[3])) / 1e4
        trailing_adv = float(row.get("trailing_adv") or np.nan)
        participation = ((RESEARCH_NAV_USD * POSITION_WEIGHT) / trailing_adv
                         if np.isfinite(trailing_adv) and trailing_adv > 0 else np.nan)
        if not np.isfinite(participation):
            return {"deployable": False, "reason": "liquidity unavailable"}
        impact_bps = IMPACT_COEFFICIENT_BPS * math.sqrt(max(participation, 0.0))
        nonspread_per_side = ADVERSE_SELECTION_BPS_PER_SIDE + impact_bps
        nonspread_round_trip = 2 * nonspread_per_side / 1e4 * (1 + beta)
        borrow_return = (borrow_fee * HOLDING_SESSIONS / 252 / 1e4
                         if side < 0 else 0.0)
        all_in = embedded_spread + nonspread_round_trip + borrow_return
    else:
        embedded_spread = 0.0
        participation = None
        impact_bps = None
        all_in = 2 * BAR_COST_BPS_PER_SIDE / 1e4 * (1 + beta)
        if side < 0:
            all_in += borrow_fee * HOLDING_SESSIONS / 252 / 1e4
    if doubled:
        all_in *= 2.0
    ledger_round_trip = max(all_in - embedded_spread, 0.0)
    denominator = 2 * (1 + beta)
    return {
        "deployable": True, "mode": "NBBO" if quote_mode else "BAR",
        "all_in_cost": float(all_in),
        "ledger_cost_bps_per_side": float(ledger_round_trip * 1e4 / denominator),
        "embedded_spread_bps": float(embedded_spread * 1e4),
        "participation_rate": participation, "impact_bps_per_side": impact_bps,
        "borrow_fee_bps_annual": borrow_fee if side < 0 else 0.0,
    }
