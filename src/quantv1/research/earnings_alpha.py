"""Elastic-net-first earnings alpha sprint and hard promotion gates.

This module does not acquire data. It builds leak-free features from staged
event windows, compares a price-only elastic net with a structured-earnings
elastic net, simulates quote-side execution under capital constraints, and
evaluates the frozen promotion gates in ``docs/strategy.md``.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import duckdb
from scipy.stats import kurtosis, norm, skew, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import DATA_DIR, ROOT
from ..db import connect
from ..ingest.earnings import (
    PROTOCOL_LOCK_DATE,
    RETROSPECTIVE_HOLDOUT_START,
    SAMPLE_END,
    SPRINT_VERSION,
    VALIDATION_START,
)
from ..portfolio.ledger import MarkedExposureBook, PortfolioLedger
from .protocol import (
    BETA_VERSION,
    MIN_PERMUTATION_FRACTION,
    TARGET_VERSION,
    clustered_mean_ci,
    clustered_portfolio_bootstrap,
    execution_cost_estimate,
    first_session_after,
    hac_statistics,
    power_requirements,
    shrink_and_clip_beta,
    trading_sessions,
)
from .earnings_strategy import (
    COST_HURDLE_MULTIPLE,
    decision_from_prediction,
)

FEATURE_PATH = DATA_DIR / "earnings_features.parquet"
FEATURE_METADATA_PATH = DATA_DIR / "earnings_features_metadata.json"
REPORT_PATH = DATA_DIR / "earnings_alpha_report.json"
SPEC_LOCK_PATH = DATA_DIR / "earnings_model_spec_lock.json"
RETROSPECTIVE_REPORT_PATH = DATA_DIR / "earnings_retrospective_holdout_report.json"
# Coarse V5 screen: bar execution with deliberately punitive assumed costs.
# Historical NBBO remains a later promotion requirement, not an early blocker.
DECISION_MINUTES = 30
DELAYED_MINUTES = 60
TARGET_TRADING_DAYS = 5
BAR_COST_BPS_PER_SIDE = 15.0
MIN_TRAIN = 200
MIN_VALIDATION = 100
MAX_CONCURRENT = 5
POSITION_WEIGHT = 0.05
MAX_GROSS = 0.25
MAX_SECTOR_GROSS = 0.15
MAX_NET = 0.15
MIN_QUOTE_COVERAGE = 0.95
EULER = 0.5772156649
COARSE_SAMPLE_MODULUS = 4
COARSE_SAMPLE_REMAINDER = 0
ARTIFACT_VERSION = "earnings-features-v2"

PRICE_NUMERIC = [
    "gap", "reaction_1m", "reaction_5m", "reaction_30m",
    "pre_event_volatility", "trailing_adv", "first5_volume_ratio",
    "abnormal_volume_30m",
]
PRICE_CATEGORICAL = ["release_session", "sector"]
EARNINGS_NUMERIC = [
    "eps_surprise", "revenue_surprise", "eps_analyst_count",
    "revenue_analyst_count", "has_point_in_time_consensus",
    "gross_margin_surprise", "free_cash_flow_surprise", "bookings_surprise",
    "guidance_eps_surprise", "guidance_revenue_surprise", "has_guidance",
]
EARNINGS_CATEGORICAL = ["fiscal_quarter"]
POSITIONING_NUMERIC = [
    "implied_move", "implied_volatility", "days_to_cover",
    "institutional_ownership", "passive_ownership",
]
STRUCTURED_COVERAGE_MIN = 0.80
MIN_REPRESENTATIVE_GROUP_EVENTS = 20
STRUCTURED_NUMERIC = [
    "eps_surprise_z", "revenue_surprise_z", "guidance_surprise_z",
    "analyst_dispersion_z", "revision_breadth", "pre_event_volatility",
    "liquidity",
]


class EarningsStudyError(RuntimeError):
    pass


@lru_cache(maxsize=64)
def _read_frame_cached(path: str) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        return con.execute("SELECT * FROM read_parquet(?)", [path]).df() \
            .sort_values("ts").reset_index(drop=True)
    finally:
        con.close()


def _read_frame(path: str | None) -> pd.DataFrame:
    if path is None or pd.isna(path) or not Path(str(path)).exists():
        return pd.DataFrame()
    return _read_frame_cached(str(path))


def _to_utc(series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def _rth(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    utc = _to_utc(result["ts"])
    eastern = utc.dt.tz_convert("America/New_York")
    minutes = eastern.dt.hour * 60 + eastern.dt.minute
    result["local_date"] = eastern.dt.date
    return result[(minutes >= 570) & (minutes < 960)].copy()


def _decision_anchor(bars: pd.DataFrame, public_time, release_session: str):
    regular = _rth(bars)
    if regular.empty:
        return None, None
    public = pd.Timestamp(public_time, tz="UTC") if pd.Timestamp(public_time).tzinfo is None \
        else pd.Timestamp(public_time).tz_convert("UTC")
    public_local_date = public.tz_convert("America/New_York").date()
    ts = _to_utc(regular["ts"])
    if release_session == "AMC":
        candidates = regular[regular["local_date"] > public_local_date]
    elif release_session == "BMO":
        candidates = regular[regular["local_date"] >= public_local_date]
    else:
        candidates = regular[ts > public]
    if candidates.empty:
        return None, None
    session_date = candidates.iloc[0]["local_date"]
    session = regular[regular["local_date"] == session_date]
    session_ts = _to_utc(session["ts"])
    anchor = max(session_ts.iloc[0], public) if release_session == "DURING" else session_ts.iloc[0]
    return anchor, session


def _quote_at(quotes: pd.DataFrame, timestamp, *, after: bool = True,
              tolerance_minutes: int = 2):
    if quotes.empty:
        return None
    ts = _to_utc(quotes["ts"])
    target = pd.Timestamp(timestamp)
    mask = ts >= target if after else ts <= target
    candidates = quotes[mask]
    if candidates.empty:
        return None
    row = candidates.iloc[0] if after else candidates.iloc[-1]
    row_ts = pd.Timestamp(row["ts"], tz="UTC") if pd.Timestamp(row["ts"]).tzinfo is None \
        else pd.Timestamp(row["ts"]).tz_convert("UTC")
    distance = abs((row_ts - target).total_seconds())
    bid, ask = row.get("bid"), row.get("ask")
    if distance > tolerance_minutes * 60 or not np.isfinite(bid) or not np.isfinite(ask):
        return None
    if bid <= 0 or ask < bid:
        return None
    return {"ts": row_ts, "bid": float(bid), "ask": float(ask),
            "mid": float((bid + ask) / 2)}


def _known_bar_close(bars: pd.DataFrame, timestamp) -> float | None:
    """Last close fully known at timestamp (bar start must be strictly earlier)."""
    if bars.empty:
        return None
    ts = _to_utc(bars["ts"])
    index = int(np.searchsorted(ts.to_numpy(dtype="datetime64[ns]").astype("int64"),
                                pd.Timestamp(timestamp).value, side="left")) - 1
    if index < 0:
        return None
    value = bars.iloc[index].get("close")
    return float(value) if value is not None and np.isfinite(value) and value > 0 else None


def _next_bar_open(bars: pd.DataFrame, timestamp) -> dict | None:
    """First one-minute bar beginning at or after a completed decision window."""
    if bars.empty:
        return None
    ts = _to_utc(bars["ts"])
    index = int(np.searchsorted(ts.to_numpy(dtype="datetime64[ns]").astype("int64"),
                                pd.Timestamp(timestamp).value, side="left"))
    if index >= len(bars):
        return None
    value = bars.iloc[index].get("open")
    if value is None or not np.isfinite(value) or value <= 0:
        return None
    return {"time": ts.iloc[index], "price": float(value)}


def _five_day_exit(asset_bars: pd.DataFrame, benchmark_bars: pd.DataFrame,
                   session_date: date) -> dict | None:
    """Fifth subsequent common RTH close on the same Polygon adjustment basis."""
    asset = _rth(asset_bars)
    benchmark = _rth(benchmark_bars)
    asset_dates = set(asset["local_date"])
    benchmark_dates = set(benchmark["local_date"])
    common_dates = sorted(asset_dates & benchmark_dates)
    future_dates = [day for day in common_dates if day > session_date]
    if len(future_dates) < TARGET_TRADING_DAYS:
        return None
    exit_date = future_dates[TARGET_TRADING_DAYS - 1]
    asset_close = asset.loc[asset["local_date"] == exit_date, "close"].iloc[-1]
    benchmark_close = benchmark.loc[
        benchmark["local_date"] == exit_date, "close"
    ].iloc[-1]
    if not all(np.isfinite(value) and value > 0 for value in
               (asset_close, benchmark_close)):
        return None
    mark_dates = [day for day in common_dates
                  if session_date <= day <= exit_date]
    marks = [{
        "date": str(day),
        "asset_close": float(asset.loc[
            asset["local_date"] == day, "close"
        ].iloc[-1]),
        "benchmark_close": float(benchmark.loc[
            benchmark["local_date"] == day, "close"
        ].iloc[-1]),
    } for day in mark_dates]
    if any(not all(np.isfinite(record[key]) and record[key] > 0
                       for key in ("asset_close", "benchmark_close"))
           for record in marks):
        return None
    return {"date": exit_date, "asset_close": float(asset_close),
            "benchmark_close": float(benchmark_close), "marks": marks}


def _prior_rth_close(bars: pd.DataFrame, session_date: date) -> float | None:
    regular = _rth(bars)
    prior = regular[regular["local_date"] < session_date]
    if prior.empty:
        return None
    value = prior.iloc[-1].get("close")
    return float(value) if np.isfinite(value) and value > 0 else None


def _daily_context(con, ticker: str, benchmark: str, session_date: date) -> dict:
    asset = con.execute("""
        SELECT date,close,volume FROM prices
        WHERE ticker=? AND date<? ORDER BY date DESC LIMIT 65
    """, [ticker, session_date]).df().sort_values("date")
    bench = con.execute("""
        SELECT date,close FROM prices
        WHERE ticker=? AND date<? ORDER BY date DESC LIMIT 65
    """, [benchmark, session_date]).df().sort_values("date")
    if asset.empty:
        return {"prior_close": None, "pre_event_volatility": None,
                "trailing_adv": None, "beta": None, "beta_raw": None,
                "beta_observations": 0, "beta_estimation_end": None,
                "beta_version": BETA_VERSION}
    prior_close = float(asset.iloc[-1]["close"])
    returns = asset["close"].pct_change().dropna()
    volatility = float(returns.tail(20).std()) if len(returns) >= 10 else None
    adv = float((asset["close"] * asset["volume"]).tail(20).mean())
    merged = asset[["date", "close"]].merge(bench, on="date", suffixes=("_asset", "_bench"))
    ar = merged["close_asset"].pct_change()
    br = merged["close_bench"].pct_change()
    valid = ar.notna() & br.notna()
    variance = br[valid].var()
    beta_observations = int(valid.sum())
    beta_raw = (float(ar[valid].cov(br[valid]) / variance)
                if beta_observations >= 20 and variance > 0 else None)
    beta = (shrink_and_clip_beta(beta_raw, beta_observations)
            if beta_raw is not None else None)
    return {"prior_close": prior_close, "pre_event_volatility": volatility,
            "trailing_adv": adv, "beta": beta, "beta_raw": beta_raw,
            "beta_observations": beta_observations,
            "beta_estimation_end": str(asset.iloc[-1]["date"]),
            "beta_version": BETA_VERSION}


def _consensus_actuals(con, event_id: str, event_time, decision_time) -> dict:
    result = {
        "eps_surprise": np.nan, "revenue_surprise": np.nan,
        "eps_surprise_raw": np.nan, "revenue_surprise_raw": np.nan,
        "eps_analyst_count": np.nan, "revenue_analyst_count": np.nan,
        "has_point_in_time_consensus": 0.0,
        "has_eps_consensus": 0.0, "has_revenue_consensus": 0.0,
        "analyst_dispersion_raw": np.nan, "revision_breadth": np.nan,
        "gross_margin_surprise": np.nan, "free_cash_flow_surprise": np.nan,
        "bookings_surprise": np.nan, "guidance_eps_surprise": np.nan,
        "guidance_revenue_surprise": np.nan, "has_guidance": 0.0,
        "guidance_surprise_raw": np.nan, "guidance_status": "MISSING_DATA",
        "guidance_eps_vs_prior": np.nan,
        "guidance_eps_vs_consensus": np.nan,
        "guidance_revenue_vs_prior": np.nan,
        "guidance_revenue_vs_consensus": np.nan,
        "implied_move": np.nan, "implied_volatility": np.nan,
        "days_to_cover": np.nan, "institutional_ownership": np.nan,
        "passive_ownership": np.nan, "borrow_available": None,
        "borrow_fee_bps_annual": np.nan, "borrow_known_at": None,
    }
    for metric, prefix in (("diluted_eps", "eps"), ("revenue", "revenue"),
                           ("gross_margin", "gross_margin"),
                           ("free_cash_flow", "free_cash_flow"),
                           ("bookings", "bookings")):
        estimate = con.execute("""
            SELECT estimate_value,analyst_count,forecast_dispersion,revision_breadth
            FROM earnings_consensus_snapshots
            WHERE earnings_event_id=? AND metric=? AND is_point_in_time=TRUE
              AND is_final_revised=FALSE AND known_at<?
            ORDER BY known_at DESC LIMIT 1
        """, [event_id, metric, decision_time]).fetchone()
        actual = con.execute("""
            SELECT actual_value FROM earnings_actuals
            WHERE earnings_event_id=? AND metric=? AND known_at<=?
            ORDER BY known_at DESC LIMIT 1
        """, [event_id, metric, decision_time]).fetchone()
        if estimate and actual:
            scale = max(abs(float(estimate[0])), 0.01 if metric == "diluted_eps" else 1.0)
            surprise = (float(actual[0]) - float(estimate[0])) / scale
            result[f"{prefix}_surprise"] = surprise
            result[f"{prefix}_surprise_raw"] = surprise
            if f"{prefix}_analyst_count" in result:
                result[f"{prefix}_analyst_count"] = estimate[1]
            if metric == "diluted_eps":
                result["analyst_dispersion_raw"] = estimate[2]
                result["revision_breadth"] = estimate[3]
                result["has_eps_consensus"] = 1.0
            if metric == "revenue":
                result["has_revenue_consensus"] = 1.0
            result["has_point_in_time_consensus"] = 1.0
    guidance_composites = []
    for metric, output, prefix in (
            ("guidance_eps", "guidance_eps_surprise", "guidance_eps"),
            ("guidance_revenue", "guidance_revenue_surprise", "guidance_revenue")):
        guidance = con.execute("""
            SELECT (lower_value+upper_value)/2,guidance_status
            FROM earnings_guidance_snapshots
            WHERE earnings_event_id=? AND metric=? AND guidance_role='new'
              AND known_at<=? AND guidance_status='AVAILABLE'
            ORDER BY known_at DESC LIMIT 1
        """, [event_id, metric, decision_time]).fetchone()
        previous = con.execute("""
            SELECT (lower_value+upper_value)/2 FROM earnings_guidance_snapshots
            WHERE earnings_event_id=? AND metric=? AND guidance_role='previous'
              AND guidance_status='AVAILABLE' AND known_at<?
            ORDER BY known_at DESC LIMIT 1
        """, [event_id, metric, event_time]).fetchone()
        consensus = con.execute("""
            SELECT estimate_value FROM earnings_consensus_snapshots
            WHERE earnings_event_id=? AND metric=? AND is_point_in_time=TRUE
              AND is_final_revised=FALSE AND known_at<?
            ORDER BY known_at DESC LIMIT 1
        """, [event_id, metric, event_time]).fetchone()
        comparisons = []
        if guidance and guidance[0] is not None:
            new_value = float(guidance[0])
            if previous and previous[0] is not None:
                prior_value = float(previous[0])
                value = (new_value - prior_value) / max(abs(prior_value), 0.01)
                result[f"{prefix}_vs_prior"] = value
                comparisons.append(value)
            if consensus and consensus[0] is not None:
                consensus_value = float(consensus[0])
                value = ((new_value - consensus_value) /
                         max(abs(consensus_value), 0.01))
                result[f"{prefix}_vs_consensus"] = value
                comparisons.append(value)
        if comparisons:
            result[output] = float(np.mean(comparisons))
            guidance_composites.append(result[output])
            result["has_guidance"] = 1.0
            result["guidance_status"] = "AVAILABLE"
        elif con.execute("""
            SELECT 1 FROM earnings_guidance_snapshots
            WHERE earnings_event_id=? AND metric=? AND guidance_status='NO_GUIDANCE'
              AND known_at<=?
            LIMIT 1
        """, [event_id, metric, decision_time]).fetchone():
            result["guidance_status"] = "NO_GUIDANCE"
    if guidance_composites:
        result["guidance_surprise_raw"] = float(np.mean(guidance_composites))
    result["has_point_in_time_consensus"] = float(
        result["has_eps_consensus"] and result["has_revenue_consensus"]
    )
    options = con.execute("""
        SELECT implied_move,implied_volatility FROM earnings_options_expectations
        WHERE earnings_event_id=? AND observed_at<? ORDER BY observed_at DESC LIMIT 1
    """, [event_id, event_time]).fetchone()
    if options:
        result["implied_move"], result["implied_volatility"] = options
    positioning = con.execute("""
        SELECT days_to_cover,institutional_ownership,passive_ownership,
               borrow_available,borrow_fee_bps_annual,borrow_known_at
        FROM earnings_positioning_snapshots
        WHERE earnings_event_id=? AND observed_at<? ORDER BY observed_at DESC LIMIT 1
    """, [event_id, event_time]).fetchone()
    if positioning:
        (result["days_to_cover"], result["institutional_ownership"],
         result["passive_ownership"], result["borrow_available"],
         result["borrow_fee_bps_annual"], result["borrow_known_at"]) = positioning
    return result


def _window_features(con, row) -> dict | None:
    bars = _read_frame(row.bars_path)
    benchmark_bars = _read_frame(row.benchmark_bars_path)
    quotes = _read_frame(row.quotes_path)
    benchmark_quotes = _read_frame(row.benchmark_quotes_path)
    if bars.empty or benchmark_bars.empty:
        return None
    anchor, session = _decision_anchor(bars, row.earliest_public_time, row.release_session)
    if anchor is None or session is None or session.empty:
        return None
    session_date = session.iloc[0]["local_date"]
    decision = anchor + pd.Timedelta(minutes=DECISION_MINUTES)
    delayed = anchor + pd.Timedelta(minutes=DELAYED_MINUTES)
    entry = _next_bar_open(session, decision)
    delayed_entry = _next_bar_open(session, delayed)
    benchmark_regular = _rth(benchmark_bars)
    benchmark_session = benchmark_regular[
        benchmark_regular["local_date"] == session_date
    ]
    if entry is None or delayed_entry is None or benchmark_session.empty:
        return None
    benchmark_entry = _next_bar_open(benchmark_session, entry["time"])
    benchmark_delayed = _next_bar_open(benchmark_session, delayed_entry["time"])
    benchmark_event_open = _next_bar_open(benchmark_session, anchor)
    if benchmark_entry is None or benchmark_delayed is None or benchmark_event_open is None:
        return None
    exit_values = _five_day_exit(bars, benchmark_bars, session_date)
    if exit_values is None:
        return None
    exit_time = pd.Timestamp(f"{exit_values['date']} 16:00",
                             tz="America/New_York").tz_convert("UTC")
    entry_quote = _quote_at(quotes, decision)
    delayed_entry_quote = _quote_at(quotes, delayed)
    exit_quote = _quote_at(quotes, exit_time, after=False)
    benchmark_entry_quote = (
        _quote_at(benchmark_quotes, entry_quote["ts"])
        if entry_quote else None
    )
    benchmark_delayed_quote = (
        _quote_at(benchmark_quotes, delayed_entry_quote["ts"])
        if delayed_entry_quote else None
    )
    benchmark_exit_quote = _quote_at(
        benchmark_quotes, exit_time, after=False
    )
    quote_complete = all(value is not None for value in (
        entry_quote, delayed_entry_quote, exit_quote,
        benchmark_entry_quote, benchmark_delayed_quote, benchmark_exit_quote,
    ))

    context = _daily_context(con, row.ticker, row.benchmark_ticker, session_date)
    session_open = float(session.iloc[0]["open"])
    session_ts = _to_utc(session["ts"])
    anchor_index = int(np.searchsorted(
        session_ts.to_numpy(dtype="datetime64[ns]").astype("int64"),
                                       anchor.value, side="left"))
    event_open = float(session.iloc[anchor_index]["open"]) if anchor_index < len(session) else None
    p1 = _known_bar_close(session, anchor + pd.Timedelta(minutes=1))
    p30 = _known_bar_close(session, decision)
    prior_close = _prior_rth_close(bars, session_date)
    gap = session_open / prior_close - 1 if prior_close else np.nan
    reaction1 = p1 / event_open - 1 if p1 and event_open else np.nan
    reaction5_price = _known_bar_close(session, anchor + pd.Timedelta(minutes=5))
    reaction5 = reaction5_price / event_open - 1 if reaction5_price and event_open else np.nan
    reaction30 = p30 / event_open - 1 if p30 and event_open else np.nan
    benchmark_p30 = _known_bar_close(benchmark_session, decision)
    benchmark_reaction30 = (
        benchmark_p30 / benchmark_event_open["price"] - 1
        if benchmark_p30 and benchmark_event_open["price"] else np.nan
    )
    first5 = session[(session_ts >= anchor) &
                     (session_ts < anchor + pd.Timedelta(minutes=5))]
    expected_minute_volume = ((context["trailing_adv"] / prior_close / 390)
                              if context["trailing_adv"] and prior_close else None)
    volume_ratio = (float(first5["volume"].sum()) / (expected_minute_volume * 5)
                    if expected_minute_volume and not first5.empty else np.nan)
    first30 = session[(session_ts >= anchor) & (session_ts < decision)]
    abnormal_volume30 = (
        float(first30["volume"].sum()) / (expected_minute_volume * DECISION_MINUTES)
        if expected_minute_volume and not first30.empty else np.nan
    )

    if quote_complete:
        target_entry = entry_quote["mid"]
        target_delayed_entry = delayed_entry_quote["mid"]
        target_exit = exit_quote["mid"]
        target_benchmark_entry = benchmark_entry_quote["mid"]
        target_benchmark_delayed_entry = benchmark_delayed_quote["mid"]
        target_benchmark_exit = benchmark_exit_quote["mid"]
        target_price_basis = "NBBO_MID"
    else:
        target_entry = entry["price"]
        target_delayed_entry = delayed_entry["price"]
        target_exit = exit_values["asset_close"]
        target_benchmark_entry = benchmark_entry["price"]
        target_benchmark_delayed_entry = benchmark_delayed["price"]
        target_benchmark_exit = exit_values["benchmark_close"]
        target_price_basis = "BAR_MID_PROXY"
    raw_5d = target_exit / target_entry - 1
    benchmark_5d = target_benchmark_exit / target_benchmark_entry - 1
    sector_residual_5d = raw_5d - benchmark_5d
    beta = context["beta"]
    reaction30_residual = (reaction30 - beta * benchmark_reaction30
                           if beta is not None and np.isfinite(benchmark_reaction30)
                           else np.nan)
    reaction30_score = (reaction30_residual / context["pre_event_volatility"]
                        if np.isfinite(reaction30_residual) and
                        context["pre_event_volatility"] and context["pre_event_volatility"] > 0
                        else np.nan)
    beta_hedged_5d = (raw_5d - beta * benchmark_5d
                      if beta is not None else np.nan)
    delayed_raw_5d = target_exit / target_delayed_entry - 1
    delayed_benchmark_5d = (target_benchmark_exit /
                            target_benchmark_delayed_entry - 1)
    delayed_sector_residual_5d = delayed_raw_5d - delayed_benchmark_5d
    delayed_beta_hedged_5d = (delayed_raw_5d - beta * delayed_benchmark_5d
                              if beta is not None else np.nan)
    consensus = _consensus_actuals(con, row.earnings_event_id,
                                   row.earliest_public_time,
                                   decision.to_pydatetime())
    # A validation event whose exit reaches the sealed retrospective period
    # would leak holdout prices into model selection. Keep it out of both cells.
    if session_date >= RETROSPECTIVE_HOLDOUT_START:
        time_bucket = "RETROSPECTIVE_HOLDOUT_TIME"
    elif exit_values["date"] >= RETROSPECTIVE_HOLDOUT_START:
        time_bucket = "EMBARGOED_OUTCOME"
    elif session_date < VALIDATION_START:
        time_bucket = "TRAIN_TIME"
    else:
        time_bucket = "VALIDATION_TIME"
    record = {
        "earnings_event_id": row.earnings_event_id, "ticker": row.ticker,
        "public_time": row.earliest_public_time, "entry_time": entry["time"],
        "exit_time": exit_time, "delayed_entry_time": delayed_entry["time"],
        "delayed_exit_time": exit_time, "release_session": row.release_session,
        "timestamp_status": row.timestamp_status, "fiscal_quarter": row.fiscal_quarter,
        "sector": row.sector or "Unknown", "benchmark_ticker": row.benchmark_ticker,
        "company_bucket": row.company_bucket, "time_bucket": time_bucket,
        "company_size_bucket": row.company_size_bucket or "MISSING",
        "gap": gap, "reaction_1m": reaction1, "reaction_5m": reaction5,
        "reaction_30m": reaction30,
        "residual_reaction_30m": reaction30_residual,
        "reaction_score": reaction30_score,
        "abnormal_volume_30m": abnormal_volume30,
        "pre_event_volatility": context["pre_event_volatility"],
        "trailing_adv": context["trailing_adv"],
        "first5_volume_ratio": volume_ratio, "beta": beta,
        "beta_raw": context.get("beta_raw", context["beta"]),
        "beta_observations": context.get("beta_observations", 0),
        "beta_estimation_end": context.get("beta_estimation_end"),
        "beta_version": context.get("beta_version", BETA_VERSION),
        "target_raw_5d": raw_5d,
        "target_sector_residual_5d": sector_residual_5d,
        "target_beta_hedged_5d": beta_hedged_5d,
        "target_actually_hedged_5d": beta_hedged_5d,
        "delayed_target_raw_5d": delayed_raw_5d,
        "delayed_target_sector_residual_5d": delayed_sector_residual_5d,
        "delayed_target_beta_hedged_5d": delayed_beta_hedged_5d,
        "delayed_target_actually_hedged_5d": delayed_beta_hedged_5d,
        "target_version": TARGET_VERSION,
        "target_price_basis": target_price_basis,
        "target_entry_mid": target_entry, "target_exit_mid": target_exit,
        "target_benchmark_entry_mid": target_benchmark_entry,
        "target_benchmark_exit_mid": target_benchmark_exit,
        "entry_price": entry["price"], "exit_price": exit_values["asset_close"],
        "delayed_entry_price": delayed_entry["price"],
        "benchmark_entry_price": benchmark_entry["price"],
        "benchmark_delayed_entry_price": benchmark_delayed["price"],
        "benchmark_exit_price": exit_values["benchmark_close"],
        "daily_marks": json.dumps(exit_values["marks"]),
        "entry_bid": entry_quote["bid"] if entry_quote else np.nan,
        "entry_ask": entry_quote["ask"] if entry_quote else np.nan,
        "delayed_entry_bid": (delayed_entry_quote["bid"]
                              if delayed_entry_quote else np.nan),
        "delayed_entry_ask": (delayed_entry_quote["ask"]
                              if delayed_entry_quote else np.nan),
        "exit_bid": exit_quote["bid"] if exit_quote else np.nan,
        "exit_ask": exit_quote["ask"] if exit_quote else np.nan,
        "benchmark_entry_bid": (benchmark_entry_quote["bid"]
                                if benchmark_entry_quote else np.nan),
        "benchmark_entry_ask": (benchmark_entry_quote["ask"]
                                if benchmark_entry_quote else np.nan),
        "benchmark_delayed_entry_bid": (benchmark_delayed_quote["bid"]
                                        if benchmark_delayed_quote else np.nan),
        "benchmark_delayed_entry_ask": (benchmark_delayed_quote["ask"]
                                        if benchmark_delayed_quote else np.nan),
        "benchmark_exit_bid": (benchmark_exit_quote["bid"]
                               if benchmark_exit_quote else np.nan),
        "benchmark_exit_ask": (benchmark_exit_quote["ask"]
                               if benchmark_exit_quote else np.nan),
        "execution_mode": ("NEXT_EXECUTABLE_NBBO_PLUS_ASSUMED_COST"
                           if quote_complete else
                           "NEXT_MINUTE_BAR_PLUS_ASSUMED_COST"),
        "assumed_cost_bps_per_side": BAR_COST_BPS_PER_SIDE,
        "quote_complete": quote_complete, "quote_coverage": row.quote_coverage,
    }
    record.update(consensus)
    return record


def build_feature_frame(verbose: bool = True, *, mode: str,
                        before: date = RETROSPECTIVE_HOLDOUT_START) -> pd.DataFrame:
    if mode not in {"coarse", "full"}:
        raise EarningsStudyError("feature mode must be 'coarse' or 'full'")
    includes_holdout = pd.Timestamp(before).date() > RETROSPECTIVE_HOLDOUT_START
    if includes_holdout and not SPEC_LOCK_PATH.exists():
        raise EarningsStudyError(
            "retrospective holdout features require a locked model specification"
        )
    if includes_holdout and mode != "full":
        raise EarningsStudyError("retrospective holdout requires a full artifact")
    sample_modulus = COARSE_SAMPLE_MODULUS if mode == "coarse" else None
    sample_remainder = COARSE_SAMPLE_REMAINDER
    # Apply additive schema migrations before opening the long-lived read-only
    # feature connection. This keeps existing research databases usable.
    migration = connect()
    migration.close()
    con = connect(read_only=True)
    windows = con.execute("""
        SELECT e.earnings_event_id,e.ticker,e.earliest_public_time,e.release_session,
               e.timestamp_status,e.fiscal_quarter,w.bars_path,w.quotes_path,
               w.benchmark_ticker,w.benchmark_bars_path,w.benchmark_quotes_path,
               w.quote_coverage,COALESCE(s.sector,'Unknown') sector,u.company_bucket,
               u.company_size_bucket
        FROM earnings_events e
        JOIN earnings_market_windows w USING(earnings_event_id)
        JOIN earnings_universe_snapshots u ON u.ticker=e.ticker AND u.universe_version=?
        LEFT JOIN ticker_sectors s ON s.ticker=e.ticker
        WHERE e.timestamp_status IN ('VERIFIED_EARLIEST','CONSERVATIVE_SEC_ONLY')
          AND w.status IN ('COMPLETE_QUOTES','BARS_ONLY')
          AND e.earliest_public_time < ?
        ORDER BY e.earliest_public_time,e.ticker
    """, ["earnings-alpha-v1-2021-06-30", before]).df()
    eligible_windows = len(windows)
    if sample_modulus:
        if sample_modulus < 2 or not 0 <= sample_remainder < sample_modulus:
            raise EarningsStudyError("invalid deterministic feature sample")
        windows = windows[windows["earnings_event_id"].map(
            lambda event_id: int(hashlib.sha256(event_id.encode()).hexdigest()[:8], 16) %
            sample_modulus == sample_remainder
        )].reset_index(drop=True)
    records = []
    for index, row in enumerate(windows.itertuples(index=False), 1):
        feature = _window_features(con, row)
        if feature:
            records.append(feature)
        if verbose and index % 100 == 0:
            print(f"  earnings features {index}/{len(windows)} usable={len(records)}")
    con.close()
    frame = pd.DataFrame(records)
    frame, feature_stats = add_structured_features(frame)
    metadata = {
        "artifact_version": ARTIFACT_VERSION, "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "before": str(before), "eligible_market_windows": eligible_windows,
        "selected_market_windows": len(windows), "feature_rows": len(frame),
        "sample_modulus": sample_modulus, "sample_remainder": sample_remainder,
        "target_version": TARGET_VERSION, "beta_version": BETA_VERSION,
        "protocol_lock_date": str(PROTOCOL_LOCK_DATE),
        "retrospective_holdout_start": str(RETROSPECTIVE_HOLDOUT_START),
        "includes_retrospective_holdout": includes_holdout,
        "code_hash": _git_hash(),
    }
    FEATURE_METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
    if not frame.empty:
        frame["artifact_mode"] = mode
        frame["artifact_version"] = ARTIFACT_VERSION
        parquet_con = duckdb.connect(":memory:")
        try:
            parquet_con.register("_earnings_features", frame)
            FEATURE_PATH.unlink(missing_ok=True)
            parquet_con.execute(
                f"COPY _earnings_features TO '{FEATURE_PATH}' (FORMAT PARQUET)"
            )
        finally:
            parquet_con.close()
        (DATA_DIR / "earnings_structured_feature_stats.json").write_text(
            json.dumps(feature_stats, indent=2, default=str)
        )
        write_con = connect()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        write_con.executemany("""
            INSERT INTO earnings_event_outcomes
                (earnings_event_id,horizon,entry_time,exit_time,raw_return,
                 sector_residual,actually_hedged_return,execution_status,
                 outcome_version,created_at,metadata)
            VALUES (?,'5d',?,?,?,?,?,'BAR_COST_SCREEN','earnings-5d-bars-v1',?,?)
            ON CONFLICT DO UPDATE SET
                entry_time=excluded.entry_time,exit_time=excluded.exit_time,
                raw_return=excluded.raw_return,
                sector_residual=excluded.sector_residual,
                actually_hedged_return=excluded.actually_hedged_return,
                execution_status=excluded.execution_status,
                created_at=excluded.created_at,metadata=excluded.metadata
        """, [(row.earnings_event_id, row.entry_time, row.exit_time,
                row.target_raw_5d, row.target_sector_residual_5d,
                row.target_beta_hedged_5d, now,
                json.dumps({"entry": "next one-minute bar after 30-minute decision",
                            "exit": "fifth subsequent common market close",
                            "cost_bps_per_side": BAR_COST_BPS_PER_SIDE}))
               for row in frame.itertuples(index=False)])
        write_con.close()
    else:
        FEATURE_PATH.unlink(missing_ok=True)
    return frame


def _load_feature_frame() -> pd.DataFrame:
    if not FEATURE_PATH.exists():
        return pd.DataFrame()
    con = duckdb.connect(":memory:")
    try:
        return con.execute("SELECT * FROM read_parquet(?)", [str(FEATURE_PATH)]).df()
    finally:
        con.close()


def _load_feature_metadata() -> dict:
    if not FEATURE_METADATA_PATH.exists():
        return {"artifact_version": "LEGACY", "mode": "unknown",
                "promotion_eligible": False}
    metadata = json.loads(FEATURE_METADATA_PATH.read_text())
    metadata["promotion_eligible"] = (
        metadata.get("mode") == "full" and
        metadata.get("target_version") == TARGET_VERSION and
        metadata.get("beta_version") == BETA_VERSION
    )
    return metadata


# EERM mismatch models M1/M2 are anchored on point-in-time analyst consensus.
# That data is only sold by vendors / WRDS and cannot be honestly reconstructed
# for free, so the two models are paused. The frozen protocol below is preserved
# verbatim and MUST NOT be weakened or reused; we only record the pause here so
# the block is durable and auditable. See MGRM (research/mgrm.py) for the
# zero-vendor pivot that reuses the reaction engine without analyst consensus.
M1_M2_PROGRAM_STATUS = "BLOCKED_DATA_ECONOMICALLY_INACCESSIBLE"
M1_M2_BLOCKED_REASON = (
    "Point-in-time analyst consensus, actuals, and pre-release guidance consensus "
    "for the sample window are only available through paid vendors or WRDS. We "
    "cannot recreate historical analyst expectations for free without fabricating "
    "provenance, so EERM M1 (fundamental surprise) and M2 (surprise-reaction "
    "mismatch) are paused indefinitely. The frozen EERM protocol is retained "
    "unchanged; no substitute or backfilled expectations data is admitted."
)


def structured_data_audit(frame: pd.DataFrame | None = None) -> dict:
    """Report whether licensed expectations data is ready for M1/M2."""
    data = frame.copy() if frame is not None else _load_feature_frame()
    con = connect(read_only=True)
    tables = {}
    for table in ("earnings_consensus_snapshots", "earnings_actuals",
                  "earnings_guidance_snapshots", "earnings_options_expectations"):
        tables[table] = int(con.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0])
    con.close()
    coverage = coverage_statistics(data)
    # No admissible vendor expectations exist -> the models are economically
    # blocked, not merely under-covered. Preserve the distinction explicitly.
    economically_blocked = sum(tables.values()) == 0
    program_status = (M1_M2_PROGRAM_STATUS if economically_blocked else
                      "READY" if coverage["gate_passed"] else "BLOCKED")
    return {
        "status": "READY" if coverage["gate_passed"] else "BLOCKED",
        "program_status": program_status,
        "blocked_reason": (M1_M2_BLOCKED_REASON
                           if program_status == M1_M2_PROGRAM_STATUS else None),
        "protocol_preserved": True,
        "pivot": "MGRM (Management Guidance Revision-Reaction Mismatch)",
        "feature_artifact": _load_feature_metadata(),
        "table_rows": tables, "coverage": coverage,
        "required": [
            "archived point-in-time EPS and revenue consensus",
            "analyst count, dispersion, and revision breadth",
            "timestamped actual EPS and revenue",
            "previous/new guidance and pre-release guidance consensus",
            "point-in-time company-size bucket for representativeness",
        ],
        "unsafe_substitutions_forbidden": [
            "current consensus backfilled into historical events",
            "final revised estimates", "records without known_at provenance",
        ],
    }


def calculate_fundamental_surprise(frame: pd.DataFrame) -> pd.Series:
    """Combine directional earnings information available by decision time."""
    columns = ["eps_surprise_z", "revenue_surprise_z", "guidance_surprise_z",
               "revision_breadth"]
    available = [column for column in columns if column in frame]
    if not available:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[available].apply(pd.to_numeric, errors="coerce").mean(
        axis=1, skipna=True
    )


def calculate_mismatch(fundamental_surprise: pd.Series,
                       residual_reaction_score: pd.Series) -> pd.Series:
    """Fundamental surprise less standardized beta-adjusted 30-minute reaction."""
    fundamental = pd.to_numeric(fundamental_surprise, errors="coerce")
    reaction = pd.to_numeric(residual_reaction_score, errors="coerce")
    return fundamental - reaction


def coverage_statistics(frame: pd.DataFrame, *,
                        minimum: float = STRUCTURED_COVERAGE_MIN,
                        min_group_events: int = MIN_REPRESENTATIVE_GROUP_EVENTS) -> dict:
    """Audit overall and representative point-in-time EPS+revenue coverage."""
    if frame.empty:
        return {"minimum_required": minimum, "gate_passed": False,
                "reason": "empty feature frame", "by_split": {}, "groups": {}}
    data = frame.copy()
    coverage_column = ("representative_consensus_coverage"
                       if "representative_consensus_coverage" in data
                       else "has_point_in_time_consensus")
    data["_coverage"] = pd.to_numeric(data[coverage_column], errors="coerce").fillna(0.0)
    data["year"] = pd.to_datetime(data["entry_time"], utc=True).dt.year
    by_split = {}
    for split, group in data.groupby("time_bucket", observed=True, dropna=False):
        by_split[str(split)] = {"events": len(group),
                                "coverage": float(group["_coverage"].mean())}

    groups = {}
    dimension_availability = {}
    dimensions = ("year", "sector", "company_size_bucket")
    representative_pass = True
    missing_dimensions = []
    for dimension in dimensions:
        if dimension not in data:
            groups[dimension] = []
            missing_dimensions.append(dimension)
            dimension_availability[dimension] = 0.0
            representative_pass = False
            continue
        normalized = data[dimension].astype("string").str.strip().str.upper()
        available = ~(normalized.isna() | normalized.isin(
            {"", "MISSING", "UNKNOWN", "NAN", "NONE"}
        ))
        availability = float(available.mean())
        dimension_availability[dimension] = availability
        if availability < minimum:
            missing_dimensions.append(dimension)
            representative_pass = False
        records = []
        for (split, value), group in data.groupby(
                ["time_bucket", dimension], observed=True, dropna=False):
            eligible = len(group) >= min_group_events
            value_coverage = float(group["_coverage"].mean())
            passes = (not eligible) or value_coverage >= minimum
            records.append({"split": str(split), "group": str(value),
                            "events": len(group), "coverage": value_coverage,
                            "eligible": eligible, "passes": passes})
            if eligible and not passes:
                representative_pass = False
        groups[dimension] = records

    required_splits = ("TRAIN_TIME", "VALIDATION_TIME")
    split_pass = all(
        split in by_split and by_split[split]["coverage"] >= minimum
        for split in required_splits
    )
    return {
        "minimum_required": minimum, "minimum_group_events": min_group_events,
        "by_split": by_split, "groups": groups,
        "missing_dimensions": missing_dimensions,
        "dimension_availability": dimension_availability,
        "split_coverage_passed": split_pass,
        "representative_coverage_passed": representative_pass,
        "gate_passed": split_pass and representative_pass,
    }


def add_structured_features(frame: pd.DataFrame,
                            fit_mask: pd.Series | None = None) -> tuple[pd.DataFrame, dict]:
    """Fit winsorization/scalers on training rows only and transform all rows."""
    if frame.empty:
        return frame, {}
    result = frame.copy()
    if fit_mask is None:
        fit_mask = result["time_bucket"].eq("TRAIN_TIME")
    raw_map = {
        "eps_surprise_z": "eps_surprise_raw",
        "revenue_surprise_z": "revenue_surprise_raw",
        "guidance_surprise_z": "guidance_surprise_raw",
        "analyst_dispersion_z": "analyst_dispersion_raw",
    }
    stats = {"feature_version": "structured-earnings-v1",
             "fit_rows": int(fit_mask.sum()), "coverage_min": STRUCTURED_COVERAGE_MIN,
             "features": {}}
    for output, source in raw_map.items():
        values = pd.to_numeric(result.loc[fit_mask, source], errors="coerce").dropna()
        if values.empty:
            result[output] = np.nan
            stats["features"][output] = {"coverage": 0.0, "mean": None, "std": None}
            continue
        low, high = values.quantile([0.01, 0.99])
        clipped = values.clip(low, high)
        mean, std = float(clipped.mean()), float(clipped.std(ddof=0))
        std = std if std > 1e-12 else 1.0
        all_values = pd.to_numeric(result[source], errors="coerce").clip(low, high)
        result[output] = (all_values - mean) / std
        stats["features"][output] = {
            "coverage": float(len(values) / max(int(fit_mask.sum()), 1)),
            "mean": mean, "std": std,
            "winsor_low": float(low), "winsor_high": float(high),
        }
    result["liquidity"] = np.log1p(pd.to_numeric(result["trailing_adv"], errors="coerce"))
    result["fundamental_surprise_score"] = calculate_fundamental_surprise(result)
    result["surprise_reaction_mismatch"] = calculate_mismatch(
        result["fundamental_surprise_score"], result["reaction_score"]
    )
    result["representative_consensus_coverage"] = result["has_point_in_time_consensus"]
    stats["representative_coverage"] = float(
        result.loc[fit_mask, "representative_consensus_coverage"].mean()
    ) if fit_mask.any() else 0.0
    stats["coverage"] = coverage_statistics(result)
    return result, stats


def descriptive_tables(frame: pd.DataFrame) -> dict:
    data = frame.dropna(subset=["target_beta_hedged_5d"]).copy()
    if data.empty:
        return {}
    data["initial_direction"] = np.sign(data["reaction_30m"])
    data["continuation"] = data["initial_direction"] * data["target_beta_hedged_5d"] > 0
    data["year"] = pd.to_datetime(data["entry_time"]).dt.year

    def table(columns):
        grouped = data.groupby(columns, observed=True, dropna=False)
        result = grouped.agg(
            n=("ticker", "size"),
            mean_raw_5d=("target_raw_5d", "mean"),
            mean_beta_hedged_5d=("target_beta_hedged_5d", "mean"),
            median_beta_hedged_5d=("target_beta_hedged_5d", "median"),
            continuation_rate=("continuation", "mean"),
        ).reset_index()
        return json.loads(result.to_json(orient="records"))

    def decile(column: str) -> list[dict]:
        available = data.dropna(subset=[column]).copy()
        if len(available) < 20 or available[column].nunique() < 3:
            return []
        bins = min(10, available[column].nunique(), len(available) // 5)
        available["decile"] = pd.qcut(available[column], bins, labels=False,
                                      duplicates="drop")
        grouped = available.groupby("decile", observed=True).agg(
            n=("ticker", "size"),
            mean_signal=(column, "mean"),
            mean_raw_5d=("target_raw_5d", "mean"),
            mean_beta_hedged_5d=("target_beta_hedged_5d", "mean"),
            median_beta_hedged_5d=("target_beta_hedged_5d", "median"),
        ).reset_index()
        return json.loads(grouped.to_json(orient="records"))
    return {
        "reaction_30m_deciles": decile("reaction_30m"),
        "eps_surprise_deciles": decile("eps_surprise_z"),
        "revenue_surprise_deciles": decile("revenue_surprise_z"),
        "guidance_surprise_deciles": decile("guidance_surprise_z"),
        "surprise_reaction_mismatch_deciles": decile("surprise_reaction_mismatch"),
        "timestamp_tier": table(["timestamp_status"]),
        "release_session": table(["release_session"]),
        "year": table(["year"]),
        "sector": table(["sector"]),
        "consensus_coverage": float(data["has_point_in_time_consensus"].mean()),
    }


def _pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    features = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    return Pipeline([("features", features),
                     ("model", ElasticNet(max_iter=20_000, random_state=23))])


def purged_group_time_splits(train: pd.DataFrame, n_splits: int,
                             embargo_days: int = 20) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward CV with max-horizon embargo and atomic earnings-event groups.

    Company leakage is tested independently by the frozen unseen-company bucket;
    removing every recurring panel company from each time fold would leave no
    training panel. Duplicate horizons/transcript rows from one event are kept
    atomic here.
    """
    ordered = train.reset_index(drop=True)
    times = pd.to_datetime(ordered["entry_time"], utc=True)
    n = len(ordered)
    first_validation = max(1, n // (n_splits + 1))
    boundaries = np.linspace(first_validation, n, n_splits + 1, dtype=int)
    folds = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        validation = np.arange(start, end)
        if not len(validation):
            continue
        cutoff = times.iloc[start] - pd.Timedelta(days=embargo_days)
        group_column = "earnings_event_id" if "earnings_event_id" in ordered else None
        validation_groups = (set(ordered.iloc[validation][group_column])
                             if group_column else set(validation))
        training = np.asarray([
            index for index in range(start)
            if times.iloc[index] < cutoff and
            (not group_column or ordered.iloc[index][group_column] not in validation_groups)
        ])
        if len(training) >= 20:
            folds.append((training, validation))
    if len(folds) < 2:
        raise EarningsStudyError("not enough observations for purged grouped time CV")
    return folds


def _fit(train: pd.DataFrame, numeric: list[str], categorical: list[str]):
    splits = min(5, max(2, len(train) // 100))
    cv = purged_group_time_splits(train, splits, embargo_days=20)
    search = GridSearchCV(
        _pipeline(numeric, categorical),
        {"model__alpha": np.logspace(-6, -2, 9),
         "model__l1_ratio": [0.1, 0.5, 0.9]},
        scoring="neg_mean_squared_error", cv=cv, n_jobs=1,
    )
    search.fit(train[numeric + categorical], train["target_beta_hedged_5d"])
    return search.best_estimator_, search.best_params_


def _execution_prices(row: dict, side: int, delayed: bool = False) -> dict | None:
    hedge_side = -side if row["beta"] >= 0 else side
    quote_fields = {
        "entry": ("delayed_entry_ask" if side > 0 else "delayed_entry_bid")
        if delayed else ("entry_ask" if side > 0 else "entry_bid"),
        "exit": "exit_bid" if side > 0 else "exit_ask",
        "benchmark_entry": (
            "benchmark_delayed_entry_ask" if hedge_side > 0
            else "benchmark_delayed_entry_bid"
        ) if delayed else (
            "benchmark_entry_ask" if hedge_side > 0 else "benchmark_entry_bid"
        ),
        "benchmark_exit": (
            "benchmark_exit_bid" if hedge_side > 0 else "benchmark_exit_ask"
        ),
    }
    quote_values = {name: row.get(field) for name, field in quote_fields.items()}
    if (bool(row.get("quote_complete", False)) and
            all(value is not None and np.isfinite(value) and value > 0
                for value in quote_values.values())):
        return {**{name: float(value) for name, value in quote_values.items()},
                "mode": "NBBO"}
    values = {
        "entry": row["delayed_entry_price"] if delayed else row["entry_price"],
        "exit": row["exit_price"],
        "benchmark_entry": (row["benchmark_delayed_entry_price"] if delayed
                            else row["benchmark_entry_price"]),
        "benchmark_exit": row["benchmark_exit_price"],
    }
    if not all(np.isfinite(value) and value > 0 for value in values.values()):
        return None
    return {**{name: float(value) for name, value in values.items()}, "mode": "BAR"}


def _bar_leg(row, side: int, cost: dict, delayed: bool = False) -> float | None:
    prices = _execution_prices(row, side, delayed)
    if prices is None:
        return None
    entry = prices["entry"]
    benchmark_entry = prices["benchmark_entry"]
    exit_price = prices["exit"]
    benchmark_exit = prices["benchmark_exit"]
    beta = row["beta"]
    prices_to_check = [entry, exit_price, benchmark_entry, benchmark_exit]
    if (not all(np.isfinite(value) and value > 0 for value in prices_to_check) or
            not np.isfinite(beta)):
        return None
    asset = side * (exit_price / entry - 1)
    hedge_side = -side if beta >= 0 else side
    benchmark = hedge_side * (benchmark_exit / benchmark_entry - 1)
    round_trip_cost = cost["ledger_cost_bps_per_side"] * 2 / 1e4
    return float(asset + abs(beta) * benchmark -
                 round_trip_cost * (1 + abs(beta)))


def simulate_portfolio(frame: pd.DataFrame, predictions: np.ndarray, *,
                       delayed: bool = False, doubled_costs: bool = False) -> dict:
    data = frame.copy()
    data["prediction"] = predictions
    data = data.sort_values("delayed_entry_time" if delayed else "entry_time")
    risk_book = MarkedExposureBook(cost_bps_per_side=0.0)
    risk_rejections = {"max_positions": 0, "duplicate_ticker": 0,
                       "gross": 0, "sector_gross": 0, "net": 0,
                       "cost_or_borrow": 0}
    trades = []
    for row in data.to_dict(orient="records"):
        entry_time = pd.Timestamp(row["delayed_entry_time"] if delayed else row["entry_time"])
        exit_time = pd.Timestamp(row["delayed_exit_time"] if delayed else row["exit_time"])
        risk_book.advance(entry_time)
        if len(risk_book.active) >= MAX_CONCURRENT:
            risk_rejections["max_positions"] += 1
            continue
        if risk_book.has_ticker(row["ticker"]):
            risk_rejections["duplicate_ticker"] += 1
            continue
        prediction = row["prediction"]
        proposed_side = 1 if prediction > 0 else -1 if prediction < 0 else 0
        if not proposed_side:
            continue
        cost = execution_cost_estimate(
            row, proposed_side, delayed=delayed, doubled=doubled_costs
        )
        if not cost["deployable"]:
            risk_rejections["cost_or_borrow"] += 1
            continue
        decision = decision_from_prediction(
            prediction, row.get("beta"), cost_bps_per_side=0.0,
            hurdle_multiple=COST_HURDLE_MULTIPLE,
            all_in_cost_estimate=cost["all_in_cost"],
        )
        if decision["side"] == 0:
            continue
        side = decision["side"]
        if not np.isfinite(row["beta"]):
            continue
        leg_return = _bar_leg(row, side, cost, delayed=delayed)
        if leg_return is None:
            continue
        execution_prices = _execution_prices(row, side, delayed)
        if execution_prices is None:
            continue
        trade = {"earnings_event_id": row["earnings_event_id"], "ticker": row["ticker"],
                 "sector": row["sector"], "release_session": row["release_session"],
                 "entry_time": str(entry_time), "exit_time": str(exit_time),
                 "side": side, "weight": POSITION_WEIGHT, "leg_return": leg_return,
                 "quarter": f"{entry_time.year}-Q{entry_time.quarter}",
                 "beta": float(row["beta"]),
                 "entry_price": execution_prices["entry"],
                 "benchmark_entry_price": execution_prices["benchmark_entry"],
                 "exit_price": execution_prices["exit"],
                 "benchmark_exit_price": execution_prices["benchmark_exit"],
                 "execution_mode": execution_prices["mode"],
                 "daily_marks": row.get("daily_marks"),
                 "announcement_date": str(pd.Timestamp(
                     row.get("public_time", entry_time)
                 ).date()),
                 "ledger_cost_bps_per_side": cost["ledger_cost_bps_per_side"],
                 "estimated_all_in_cost": cost["all_in_cost"],
                 "embedded_spread_bps": cost["embedded_spread_bps"],
                 "participation_rate": cost["participation_rate"],
                 "impact_bps_per_side": cost["impact_bps_per_side"],
                 "borrow_fee_bps_annual": cost["borrow_fee_bps_annual"],
                 "signal_hurdle_bps": decision["hurdle_bps"]}
        projection = risk_book.project(trade, entry_time)
        if projection["gross"] > MAX_GROSS + 1e-12:
            risk_rejections["gross"] += 1
            continue
        if projection["sector_gross"] > MAX_SECTOR_GROSS + 1e-12:
            risk_rejections["sector_gross"] += 1
            continue
        if abs(projection["net"]) > MAX_NET + 1e-12:
            risk_rejections["net"] += 1
            continue
        trade.update({
            "gross_exposure": ((projection["asset_notional"] +
                                projection["hedge_notional"]) / projection["nav"]),
            "entry_book_gross": projection["gross"],
            "entry_book_net": projection["net"],
            "entry_sector_gross": projection["sector_gross"],
            "pnl": projection["asset_notional"] * leg_return,
        })
        risk_book.open(trade, projection)
        trades.append(trade)
    calendar = set()
    if len(data):
        starts = pd.to_datetime(data["delayed_entry_time" if delayed else "entry_time"],
                                utc=True, errors="coerce").dropna()
        ends = pd.to_datetime(data["delayed_exit_time" if delayed else "exit_time"],
                              utc=True, errors="coerce").dropna()
        buckets = set(data.get("time_bucket", pd.Series(dtype=str)).dropna())
        if buckets == {"VALIDATION_TIME"}:
            calendar.update(trading_sessions(
                VALIDATION_START, RETROSPECTIVE_HOLDOUT_START - timedelta(days=1)
            ))
        elif buckets == {"RETROSPECTIVE_HOLDOUT_TIME"}:
            calendar.update(trading_sessions(RETROSPECTIVE_HOLDOUT_START, SAMPLE_END))
        elif len(starts) and len(ends):
            calendar.update(trading_sessions(starts.min(), ends.max()))
    portfolio = PortfolioLedger(
        cost_bps_per_side=0.0,
        calendar=calendar,
    ).run(trades)
    portfolio["risk_rejections"] = risk_rejections
    if not trades:
        return portfolio
    trade_frame = pd.DataFrame(trades)
    daily = pd.Series(portfolio["daily_returns"], dtype=float)
    std = daily.std(ddof=1)
    portfolio.update({
        "mean_pnl_bps": float(trade_frame["pnl"].mean() * 1e4),
        "hit_rate": float((trade_frame["pnl"] > 0).mean()),
        "sharpe_annual": (float(daily.mean() / std * np.sqrt(252))
                           if len(daily) > 1 and std > 0 else None),
    })
    return portfolio


def _deflated_sharpe(daily_returns: list[float], n_trials: int) -> dict:
    returns = np.asarray(daily_returns, dtype=float)
    if len(returns) < 20 or returns.std(ddof=1) == 0:
        return {"probability": None, "passes": False, "n_trials": n_trials}
    period_sharpe = returns.mean() / returns.std(ddof=1)
    g3, g4 = float(skew(returns)), float(kurtosis(returns, fisher=False))
    variance_term = 1 - g3 * period_sharpe + (g4 - 1) / 4 * period_sharpe ** 2
    standard_error = np.sqrt(variance_term / (len(returns) - 1))
    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * np.e))
    expected_max = standard_error * ((1 - EULER) * z1 + EULER * z2)
    probability = float(norm.cdf(
        (period_sharpe - expected_max) * np.sqrt(len(returns) - 1) /
        np.sqrt(variance_term)
    ))
    return {"probability": probability, "passes": probability > 0.95,
            "n_trials": n_trials}


def _metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    variable = (len(actual) > 1 and np.ptp(predicted) > 1e-12 and
                np.ptp(actual) > 1e-12)
    pearson = float(np.corrcoef(actual, predicted)[0, 1]) if variable else None
    rank = float(spearmanr(actual, predicted).statistic) if variable else None
    pearson = pearson if pearson is not None and np.isfinite(pearson) else None
    rank = rank if rank is not None and np.isfinite(rank) else None
    return {"n": len(actual), "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "mae": float(mean_absolute_error(actual, predicted)),
            "spearman_ic": rank, "pearson_ic": pearson, "ic": rank}


def _stability_and_concentration(portfolio: dict) -> dict:
    trades = pd.DataFrame(portfolio.get("trades", []))
    if trades.empty:
        return {"years": {}, "sectors": {}, "concentration": {}, "passes": False}
    trades["year"] = pd.to_datetime(trades["entry_time"], utc=True).dt.year
    years = trades.groupby("year")["pnl"].agg(["count", "sum"])
    sectors = trades.groupby("sector")["pnl"].agg(["count", "sum"])
    eligible_years = years[years["count"] >= 30]
    eligible_sectors = sectors[sectors["count"] >= 30]
    positive = trades[trades["pnl"] > 0]
    total_positive = positive["pnl"].sum()

    def max_share(column):
        if total_positive <= 0 or positive.empty:
            return None
        return float(positive.groupby(column)["pnl"].sum().max() / total_positive)
    concentration = {"company": max_share("ticker"),
                     "event_category": max_share("release_session"),
                     "quarter": max_share("quarter")}
    years_pass = len(eligible_years) > 0 and bool((eligible_years["sum"] > 0).all())
    sectors_pass = len(eligible_sectors) > 0 and float((eligible_sectors["sum"] > 0).mean()) >= 0.70
    concentration_pass = all(value is not None and value <= 0.25
                             for value in concentration.values())
    return {
        "years": json.loads(years.reset_index().to_json(orient="records")),
        "sectors": json.loads(sectors.reset_index().to_json(orient="records")),
        "concentration": concentration, "years_pass": years_pass,
        "sectors_pass": sectors_pass, "concentration_pass": concentration_pass,
        "passes": years_pass and sectors_pass and concentration_pass,
    }


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _dataset_hash(frame: pd.DataFrame) -> str:
    values = frame[["earnings_event_id", "ticker", "public_time",
                    "target_beta_hedged_5d"]] \
        .astype(str).sort_values("earnings_event_id").to_csv(index=False)
    return hashlib.sha256(values.encode()).hexdigest()[:16]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _available_model_specs(
        train: pd.DataFrame,
        validation: pd.DataFrame | None = None) -> dict[str, tuple[list[str], list[str]]]:
    specs = {"M0_price_reaction": (PRICE_NUMERIC, PRICE_CATEGORICAL)}
    gate_frame = train if validation is None else pd.concat(
        [train, validation], ignore_index=True
    )
    coverage = coverage_statistics(gate_frame)
    if coverage["gate_passed"]:
        specs["M1_structured_surprise"] = (
            _unique(PRICE_NUMERIC + STRUCTURED_NUMERIC),
            _unique(PRICE_CATEGORICAL + EARNINGS_CATEGORICAL),
        )
        specs["M2_surprise_reaction_mismatch"] = (
            _unique(PRICE_NUMERIC + STRUCTURED_NUMERIC + [
                "reaction_score", "residual_reaction_30m",
                "fundamental_surprise_score", "surprise_reaction_mismatch",
            ]),
            _unique(PRICE_CATEGORICAL + EARNINGS_CATEGORICAL),
        )
    return specs


def _grouped_permutation_order(frame: pd.DataFrame, *,
                               random_state: int = 23) -> tuple[np.ndarray, dict]:
    """Permute atomic ticker/event blocks only within year-sector strata."""
    data = frame.reset_index(drop=True).copy()
    if "year" not in data:
        if "entry_time" not in data:
            return np.arange(len(data)), {
                "status": "MISSING_YEAR", "permuted_rows": 0,
            }
        data["year"] = pd.to_datetime(data["entry_time"], utc=True).dt.year
    if "sector" not in data:
        return np.arange(len(data)), {
            "status": "MISSING_SECTOR", "permuted_rows": 0,
        }
    block_column = ("ticker" if "ticker" in data else
                    "earnings_event_id" if "earnings_event_id" in data else None)
    if block_column is None:
        return np.arange(len(data)), {
            "status": "MISSING_ATOMIC_BLOCK", "permuted_rows": 0,
        }

    rng = np.random.default_rng(random_state)
    order = np.arange(len(data))
    eligible_buckets = 0
    total_buckets = 0
    grouped = data.groupby(["year", "sector"], observed=True, dropna=False,
                           sort=True).indices
    for _, stratum_positions in grouped.items():
        positions = np.asarray(sorted(stratum_positions), dtype=int)
        blocks = {}
        for position in positions:
            blocks.setdefault(str(data.iloc[position][block_column]), []).append(position)
        by_size = {}
        for block_positions in blocks.values():
            if "entry_time" in data:
                block_positions = sorted(
                    block_positions,
                    key=lambda index: pd.Timestamp(data.iloc[index]["entry_time"]),
                )
            by_size.setdefault(len(block_positions), []).append(block_positions)
        for block_size, destination_blocks in sorted(by_size.items()):
            total_buckets += 1
            if len(destination_blocks) < 2:
                continue
            eligible_buckets += 1
            source_order = rng.permutation(len(destination_blocks))
            if np.array_equal(source_order, np.arange(len(destination_blocks))):
                source_order = np.roll(source_order, 1)
            source_blocks = [destination_blocks[index] for index in source_order]
            for destination, source in zip(destination_blocks, source_blocks):
                if len(destination) != block_size or len(source) != block_size:
                    raise EarningsStudyError("permutation block size changed")
                order[np.asarray(destination, dtype=int)] = np.asarray(source, dtype=int)

    return order, {
        "status": "COMPLETE", "strata": ["year", "sector"],
        "atomic_block": block_column, "equal_size_blocks_only": True,
        "eligible_buckets": eligible_buckets, "total_buckets": total_buckets,
        "permuted_rows": int((order != np.arange(len(data))).sum()),
    }


def permutation_controls(model: Pipeline, frame: pd.DataFrame,
                         numeric: list[str], categorical: list[str], *,
                         random_state: int = 23) -> dict:
    """Run deterministic null controls without refitting on validation data."""
    if len(frame) < 2:
        return {"status": "INSUFFICIENT_OBSERVATIONS"}
    order, grouping = _grouped_permutation_order(
        frame, random_state=random_state
    )
    features = frame[numeric + categorical].copy()
    block_permuted = features.copy()
    block_permuted.loc[:, numeric] = features[numeric].iloc[order].to_numpy()
    block_prediction = model.predict(block_permuted)
    baseline_prediction = model.predict(features)
    timestamp_prediction = baseline_prediction[order]
    target_column = ("target_beta_hedged_5d"
                     if "target_beta_hedged_5d" in frame
                     else "target_sector_residual_5d")
    actual = frame[target_column]
    return {
        "status": "COMPLETE", "random_state": random_state,
        "grouping": grouping,
        "block_feature_permutation": _metrics(actual, block_prediction),
        "shuffled_timestamp": _metrics(actual, timestamp_prediction),
    }


def _evaluate_cell(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    portfolio = simulate_portfolio(frame, prediction)
    return {
        "predictive": _metrics(frame.target_beta_hedged_5d, prediction),
        "portfolio": portfolio,
        "hac": hac_statistics(portfolio["daily_returns"]),
        "cluster_bootstrap": clustered_portfolio_bootstrap(portfolio),
        "delayed_entry": simulate_portfolio(frame, prediction, delayed=True),
        "doubled_costs": simulate_portfolio(frame, prediction, doubled_costs=True),
        "unseen_company_robustness": _metrics(
            frame.loc[frame.company_bucket == "UNSEEN_COMPANY",
                      "target_beta_hedged_5d"],
            prediction[frame.company_bucket.to_numpy() == "UNSEEN_COMPANY"],
        ) if (frame.company_bucket == "UNSEEN_COMPANY").any() else None,
    }


def _persist_experiment(result: dict, status: str) -> None:
    con = connect()
    con.execute("""
        INSERT INTO earnings_experiments
            (experiment_id,sprint_version,created_at,status,dataset_hash,code_hash,
             model_family,holdout_definition,metrics,promotion_gates)
        VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
    """, [result["experiment_id"], SPRINT_VERSION,
          datetime.now(timezone.utc).replace(tzinfo=None), status,
          result["dataset_hash"], _git_hash(), "elastic_net_5d_bar_cost",
          json.dumps(result["splits"]), json.dumps(result.get("models", {}), default=str),
          json.dumps(result.get("screening_gates", {}), default=str)])
    con.close()


def run(frame: pd.DataFrame | None = None, verbose: bool = True, *,
        lock_spec: bool = False, retrospective_holdout: bool = False,
        final_test: bool = False) -> dict:
    """Run validation or open the sealed retrospective holdout exactly once."""
    retrospective_holdout = retrospective_holdout or final_test
    if retrospective_holdout and RETROSPECTIVE_REPORT_PATH.exists():
        existing = json.loads(RETROSPECTIVE_REPORT_PATH.read_text())
        return {"status": "RETROSPECTIVE_HOLDOUT_ALREADY_OPENED", "result": existing}
    if retrospective_holdout and not SPEC_LOCK_PATH.exists():
        return {"status": "RETROSPECTIVE_HOLDOUT_BLOCKED_SPEC_NOT_LOCKED"}

    data = (frame.copy() if frame is not None else
            (_load_feature_frame() if FEATURE_PATH.exists()
             else build_feature_frame(verbose=verbose, mode="coarse")))
    if data.empty:
        result = {"status": "NOT_EVALUATED_DATA_INCOMPLETE",
                  "reason": "no Tier 1/2 earnings event bar windows"}
        REPORT_PATH.write_text(json.dumps(result, indent=2))
        return result
    data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True)
    usable = data.dropna(subset=["target_beta_hedged_5d"]) \
        .sort_values("entry_time").copy()
    train = usable[usable.time_bucket == "TRAIN_TIME"].copy()
    validation = usable[usable.time_bucket == "VALIDATION_TIME"].copy()
    holdout = usable[usable.time_bucket == "RETROSPECTIVE_HOLDOUT_TIME"].copy()
    preholdout = usable[usable.time_bucket != "RETROSPECTIVE_HOLDOUT_TIME"].copy()
    counts = {"features": len(data), "usable": len(usable), "train": len(train),
              "validation": len(validation), "retrospective_frozen": len(holdout),
              "unseen_company_validation": int(
                  (validation.company_bucket == "UNSEEN_COMPANY").sum())}
    if len(train) < MIN_TRAIN or len(validation) < MIN_VALIDATION:
        result = {
            "status": "NOT_EVALUATED_DATA_INCOMPLETE", "counts": counts,
            "required": {"train": MIN_TRAIN, "validation": MIN_VALIDATION},
            "descriptive": descriptive_tables(preholdout),
            "retrospective_holdout_outcomes_evaluated": False,
            "final_test_outcomes_evaluated": False,
        }
        REPORT_PATH.write_text(json.dumps(result, indent=2, default=str))
        return result

    dataset_hash = _dataset_hash(usable if retrospective_holdout else preholdout)
    splits = {"train_end": str(VALIDATION_START - timedelta(days=1)),
              "validation_start": str(VALIDATION_START),
              "validation_end": str(RETROSPECTIVE_HOLDOUT_START - timedelta(days=1)),
              "retrospective_holdout_start": str(RETROSPECTIVE_HOLDOUT_START),
              "protocol_lock_date": str(PROTOCOL_LOCK_DATE),
              "primary_test": "future time across all eligible companies",
              "unseen_company": "additional robustness only"}

    if retrospective_holdout:
        if holdout.empty:
            return {"status": "RETROSPECTIVE_HOLDOUT_BLOCKED_DATA_INCOMPLETE",
                    "counts": counts}
        spec = json.loads(SPEC_LOCK_PATH.read_text())
        holdout_artifact = (_load_feature_metadata() if frame is None else {
            "mode": (str(data["artifact_mode"].iloc[0])
                     if "artifact_mode" in data and len(data) else "unknown"),
            "promotion_eligible": ("artifact_mode" in data and len(data) and
                                   str(data["artifact_mode"].iloc[0]) == "full"),
            "includes_retrospective_holdout": True,
        })
        if (not holdout_artifact.get("promotion_eligible") or
                not holdout_artifact.get("includes_retrospective_holdout")):
            return {"status": "RETROSPECTIVE_HOLDOUT_BLOCKED_ARTIFACT",
                    "reason": "a full, current holdout artifact is required"}
        model_name = spec["model_name"]
        numeric, categorical = spec["numeric"], spec["categorical"]
        model = _pipeline(numeric, categorical)
        model.set_params(**spec["elastic_net_params"])
        model.fit(preholdout[numeric + categorical],
                  preholdout["target_beta_hedged_5d"])
        prediction = model.predict(holdout[numeric + categorical])
        evaluated = _evaluate_cell(holdout, prediction)
        stability = _stability_and_concentration(evaluated["portfolio"])
        dsr = _deflated_sharpe(evaluated["portfolio"]["daily_returns"],
                               int(spec["experiment_trials"]))
        result = {
            "status": "RETROSPECTIVE_HOLDOUT_OPENED", "model_name": model_name,
            "counts": counts, "splits": splits, "dataset_hash": dataset_hash,
            "experiment_id": hashlib.sha256(
                f"{SPRINT_VERSION}|{dataset_hash}|RETROSPECTIVE|{model_name}".encode()
            ).hexdigest()[:20],
            "result": evaluated, "stability": stability,
            "deflated_sharpe": dsr, "spec_lock": spec,
            "retrospective_holdout_outcomes_evaluated": True,
            "final_test_outcomes_evaluated": True,
        }
        RETROSPECTIVE_REPORT_PATH.write_text(
            json.dumps(result, indent=2, default=str)
        )
        _persist_experiment(result, result["status"])
        return result

    coverage = coverage_statistics(pd.concat([train, validation], ignore_index=True))
    model_specs = _available_model_specs(train, validation)
    structured_tested = "M1_structured_surprise" in model_specs
    models, params, validation_results, predictions = {}, {}, {}, {}
    for model_name, (numeric, categorical) in model_specs.items():
        models[model_name], params[model_name] = _fit(train, numeric, categorical)
        predictions[model_name] = models[model_name].predict(
            validation[numeric + categorical]
        )
        validation_results[model_name] = _evaluate_cell(
            validation, predictions[model_name]
        )
        pred_frame = validation[["entry_time", "sector", "target_beta_hedged_5d"]].copy()
        pred_frame["prediction"] = predictions[model_name]
        validation_results[model_name]["ic_by_year"] = {
            str(year): _metrics(group["target_beta_hedged_5d"], group["prediction"])["ic"]
            for year, group in pred_frame.groupby(pd.to_datetime(pred_frame["entry_time"]).dt.year)
        }
        validation_results[model_name]["ic_by_sector"] = {
            str(sector): _metrics(group["target_beta_hedged_5d"], group["prediction"])["ic"]
            for sector, group in pred_frame.groupby("sector") if len(group) >= 5
        }
        validation_results[model_name]["permutation_controls"] = permutation_controls(
            models[model_name], validation, numeric, categorical
        )
    best_by_rmse = min(model_specs, key=lambda name:
                       validation_results[name]["predictive"]["rmse"])
    candidate = ("M2_surprise_reaction_mismatch" if structured_tested
                 else "M0_price_reaction")
    candidate_result = validation_results[candidate]
    naive_rmse = float(np.sqrt(np.mean(
        np.square(validation["target_beta_hedged_5d"].to_numpy())
    )))
    con = connect(read_only=True)
    trials = max(32, 32 + con.execute(
        "SELECT COUNT(*) FROM earnings_experiments"
    ).fetchone()[0])
    con.close()
    stability = _stability_and_concentration(candidate_result["portfolio"])
    dsr = _deflated_sharpe(candidate_result["portfolio"]["daily_returns"], trials)
    quote_coverage = float(pd.to_numeric(
        validation.get("quote_complete", pd.Series(False, index=validation.index)),
        errors="coerce",
    ).fillna(0).mean())
    artifact = _load_feature_metadata() if frame is None else {
        "mode": (str(data["artifact_mode"].iloc[0])
                 if "artifact_mode" in data and len(data) else "unknown"),
        "promotion_eligible": ("artifact_mode" in data and len(data) and
                               str(data["artifact_mode"].iloc[0]) == "full"),
    }
    power = power_requirements(float(train["target_beta_hedged_5d"].std(ddof=1)))
    trade_frame = pd.DataFrame(candidate_result["portfolio"].get("trades", []))
    if trade_frame.empty:
        sample_audit = {"unique_executable_trades": 0, "unique_tickers": 0,
                        "announcement_dates": 0, "events_by_year": {},
                        "effective_sample_size": 0, "passes": False}
    else:
        trade_frame["year"] = pd.to_datetime(trade_frame["entry_time"], utc=True).dt.year
        events_by_year = trade_frame.groupby("year").size().to_dict()
        bootstrap_effective = candidate_result["cluster_bootstrap"].get(
            "effective_sample_size", 0
        )
        sample_audit = {
            "unique_executable_trades": int(trade_frame.earnings_event_id.nunique()),
            "unique_tickers": int(trade_frame.ticker.nunique()),
            "announcement_dates": int(trade_frame.announcement_date.nunique()),
            "events_by_year": {str(key): int(value)
                               for key, value in events_by_year.items()},
            "effective_sample_size": int(bootstrap_effective),
        }
        sample_audit["passes"] = bool(
            power.get("status") == "FROZEN" and
            sample_audit["unique_executable_trades"] >=
            power["minimum_unique_executable_trades"] and
            sample_audit["unique_tickers"] >= power["minimum_unique_tickers"] and
            sample_audit["announcement_dates"] >= power["minimum_announcement_dates"] and
            sample_audit["effective_sample_size"] >=
            power["minimum_effective_sample_size"] and
            bool(events_by_year) and min(events_by_year.values()) >=
            power["minimum_events_per_eligible_year"]
        )
    permutation_rows = candidate_result["permutation_controls"].get(
        "grouping", {}
    ).get("permuted_rows", 0)
    permutation_fraction = permutation_rows / len(validation)
    bootstrap_ci = candidate_result["cluster_bootstrap"].get(
        "confidence_intervals", {}
    )
    bootstrap_positive = all(
        bootstrap_ci.get(metric, [None])[0] is not None and
        bootstrap_ci[metric][0] > 0
        for metric in ("mean_net_return_per_trade", "total_portfolio_return",
                       "annualized_alpha")
    )
    hac_ci = candidate_result["hac"].get("annualized_alpha_ci", [None, None])
    if structured_tested:
        m1 = validation_results["M1_structured_surprise"]
        m2 = validation_results["M2_surprise_reaction_mismatch"]
        m2_loss_lift = m2["predictive"]["rmse"] < m1["predictive"]["rmse"]
        m1_ic = m1["predictive"]["spearman_ic"]
        m2_ic = m2["predictive"]["spearman_ic"]
        m2_ic_lift = (m1_ic is not None and m2_ic is not None and m2_ic > m1_ic)
        m2_net_lift = (m2["portfolio"]["net_return"] >
                       m1["portfolio"]["net_return"])
        loss_lift = (np.square(validation["target_beta_hedged_5d"] -
                               predictions["M1_structured_surprise"]) -
                     np.square(validation["target_beta_hedged_5d"] -
                               predictions["M2_surprise_reaction_mismatch"]))
        m2_m1_lift_ci = clustered_mean_ci(validation, loss_lift)
    else:
        m2_loss_lift = m2_ic_lift = m2_net_lift = False
        m2_m1_lift_ci = {"status": "NOT_TESTED",
                         "confidence_interval": [None, None]}
    screening_gates = {
        "representative_structured_coverage": coverage["gate_passed"],
        "full_feature_artifact": bool(artifact.get("promotion_eligible")),
        "m2_predictive_loss_lift_vs_m1": m2_loss_lift,
        "m2_spearman_ic_lift_vs_m1": m2_ic_lift,
        "m2_net_lift_vs_m1": m2_net_lift,
        "predictive_lift_vs_zero":
            candidate_result["predictive"]["rmse"] < naive_rmse,
        "positive_net_after_15bps_per_side":
            candidate_result["portfolio"]["net_return"] > 0,
        "net_sharpe_above_1":
            (candidate_result["portfolio"].get("sharpe_annual") is not None and
             candidate_result["portfolio"]["sharpe_annual"] > 1.0),
        "deflated_sharpe_probability_above_0_95": dsr["passes"],
        "positive_with_60min_delayed_entry":
            candidate_result["delayed_entry"]["net_return"] > 0,
        "positive_with_30bps_per_side":
            candidate_result["doubled_costs"]["net_return"] > 0,
        "stable_years_and_sectors":
            stability.get("years_pass", False) and stability.get("sectors_pass", False),
        "no_excessive_concentration": stability.get("concentration_pass", False),
        "executable_quote_coverage_at_least_95pct":
            quote_coverage >= MIN_QUOTE_COVERAGE,
        "minimum_permutation_change_coverage":
            permutation_fraction >= MIN_PERMUTATION_FRACTION,
        "power_derived_sample_requirements": sample_audit["passes"],
        "cluster_bootstrap_lower_bounds_above_zero": bootstrap_positive,
        "hac_alpha_lower_bound_above_zero":
            hac_ci[0] is not None and hac_ci[0] > 0,
        "m2_minus_m1_clustered_loss_lift_lower_bound_above_zero":
            m2_m1_lift_ci["confidence_interval"][0] is not None and
            m2_m1_lift_ci["confidence_interval"][0] > 0,
    }
    passed = all(screening_gates.values())
    experiment_id = hashlib.sha256(
        f"{SPRINT_VERSION}|{dataset_hash}|VALIDATION|{candidate}|{_git_hash()}".encode()
    ).hexdigest()[:20]
    result = {
        "status": "VALIDATION_M2_PROMOTION_PASSED" if passed
                  else "VALIDATION_M2_PROMOTION_BLOCKED",
        "experiment_id": experiment_id, "dataset_hash": dataset_hash,
        "counts": counts, "splits": splits, "candidate_model": candidate,
        "best_model_by_rmse": best_by_rmse,
        "models": validation_results, "elastic_net_params": params,
        "screening_gates": screening_gates,
        "deflated_sharpe": dsr, "stability": stability,
        "feature_artifact": artifact, "power_requirements": power,
        "sample_audit": sample_audit,
        "permutation_change_fraction": permutation_fraction,
        "m2_minus_m1_clustered_loss_lift": m2_m1_lift_ci,
        "executable_quote_coverage": quote_coverage,
        "structured_surprise_tested": structured_tested,
        "structured_coverage": coverage,
        "structured_surprise_blocker": None if structured_tested else
            "point-in-time EPS+revenue coverage is below 80% or not representative by year, sector, and company size",
        "catboost_allowed": passed,
        "descriptive": descriptive_tables(preholdout),
        "retrospective_holdout_outcomes_evaluated": False,
        "final_test_outcomes_evaluated": False,
    }
    if lock_spec:
        if not passed:
            result["specification_locked"] = False
            result["specification_lock_blocker"] = "M2 has not cleared every validation gate"
        else:
            numeric, categorical = model_specs[candidate]
            lock = {
                "model_spec_locked_at": datetime.now(timezone.utc).isoformat(),
                "validation_experiment_id": experiment_id,
                "model_name": candidate, "numeric": numeric,
                "categorical": categorical,
                "elastic_net_params": params[candidate],
                "cost_hurdle_multiple": COST_HURDLE_MULTIPLE,
                "bar_cost_bps_per_side": BAR_COST_BPS_PER_SIDE,
                "experiment_trials": trials, "splits": splits,
                "target_version": TARGET_VERSION, "beta_version": BETA_VERSION,
                "feature_artifact": artifact,
                "power_requirements": power,
                "cost_model": "observed-spread-liquidity-borrow-v1",
            }
            lock["prospective_forward_start"] = first_session_after(
                lock["model_spec_locked_at"]
            )
            if SPEC_LOCK_PATH.exists():
                raise EarningsStudyError("model specification is already locked")
            SPEC_LOCK_PATH.write_text(json.dumps(lock, indent=2, default=str))
            result["specification_locked"] = True
    _persist_experiment(result, result["status"])
    REPORT_PATH.write_text(json.dumps(result, indent=2, default=str))
    if verbose:
        print(f"Earnings five-day validation: {result['status']} "
              f"candidate={candidate} gates={screening_gates}")
    return result
