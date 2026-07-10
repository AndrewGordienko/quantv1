"""Elastic-net-first earnings alpha sprint and hard promotion gates.

This module does not acquire data. It builds leak-free features from staged
event windows, compares a price-only elastic net with a structured-earnings
elastic net, simulates quote-side execution under capital constraints, and
evaluates the frozen promotion gates in ``docs/EARNINGS_ALPHA_SPRINT.md``.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time as wall_time, timedelta, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import duckdb
from scipy.stats import kurtosis, norm, skew
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import DATA_DIR, ROOT
from ..db import connect
from ..ingest.earnings import FINAL_TEST_START, SPRINT_VERSION, VALIDATION_START
from .earnings_strategy import decision_from_prediction

FEATURE_PATH = DATA_DIR / "earnings_features.parquet"
REPORT_PATH = DATA_DIR / "earnings_alpha_report.json"
SPEC_LOCK_PATH = DATA_DIR / "earnings_model_spec_lock.json"
FINAL_REPORT_PATH = DATA_DIR / "earnings_final_test_report.json"
# Coarse V5 screen: bar execution with deliberately punitive assumed costs.
# Historical NBBO remains a later promotion requirement, not an early blocker.
DECISION_MINUTES = 30
DELAYED_MINUTES = 60
TARGET_TRADING_DAYS = 5
BAR_COST_BPS_PER_SIDE = 15.0
MIN_TRAIN = 200
MIN_VALIDATION = 100
SIGNAL_THRESHOLD_BPS = 10.0
MAX_CONCURRENT = 5
POSITION_WEIGHT = 0.05
MAX_GROSS = 0.25
MAX_SECTOR_GROSS = 0.15
MAX_NET = 0.15
MIN_QUOTE_COVERAGE = 0.95
EULER = 0.5772156649
COARSE_SAMPLE_MODULUS = 4
COARSE_SAMPLE_REMAINDER = 0

PRICE_NUMERIC = [
    "gap", "reaction_1m", "reaction_5m", "reaction_30m",
    "pre_event_volatility", "trailing_adv", "first5_volume_ratio",
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
    if not path or not Path(path).exists():
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
    asset_dates = set(asset.loc[asset["local_date"] > session_date, "local_date"])
    benchmark_dates = set(benchmark.loc[
        benchmark["local_date"] > session_date, "local_date"
    ])
    common_dates = sorted(asset_dates & benchmark_dates)
    if len(common_dates) < TARGET_TRADING_DAYS:
        return None
    exit_date = common_dates[TARGET_TRADING_DAYS - 1]
    asset_close = asset.loc[asset["local_date"] == exit_date, "close"].iloc[-1]
    benchmark_close = benchmark.loc[
        benchmark["local_date"] == exit_date, "close"
    ].iloc[-1]
    if not all(np.isfinite(value) and value > 0 for value in
               (asset_close, benchmark_close)):
        return None
    return {"date": exit_date, "asset_close": float(asset_close),
            "benchmark_close": float(benchmark_close)}


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
                "trailing_adv": None, "beta": None}
    prior_close = float(asset.iloc[-1]["close"])
    returns = asset["close"].pct_change().dropna()
    volatility = float(returns.tail(20).std()) if len(returns) >= 10 else None
    adv = float((asset["close"] * asset["volume"]).tail(20).mean())
    merged = asset[["date", "close"]].merge(bench, on="date", suffixes=("_asset", "_bench"))
    ar = merged["close_asset"].pct_change()
    br = merged["close_bench"].pct_change()
    valid = ar.notna() & br.notna()
    variance = br[valid].var()
    beta = float(ar[valid].cov(br[valid]) / variance) if valid.sum() >= 20 and variance > 0 else None
    return {"prior_close": prior_close, "pre_event_volatility": volatility,
            "trailing_adv": adv, "beta": beta}


def _consensus_actuals(con, event_id: str, event_time, decision_time) -> dict:
    result = {
        "eps_surprise": np.nan, "revenue_surprise": np.nan,
        "eps_surprise_raw": np.nan, "revenue_surprise_raw": np.nan,
        "eps_analyst_count": np.nan, "revenue_analyst_count": np.nan,
        "has_point_in_time_consensus": 0.0,
        "analyst_dispersion_raw": np.nan, "revision_breadth": np.nan,
        "gross_margin_surprise": np.nan, "free_cash_flow_surprise": np.nan,
        "bookings_surprise": np.nan, "guidance_eps_surprise": np.nan,
        "guidance_revenue_surprise": np.nan, "has_guidance": 0.0,
        "guidance_surprise_raw": np.nan, "guidance_status": "MISSING_DATA",
        "implied_move": np.nan, "implied_volatility": np.nan,
        "days_to_cover": np.nan, "institutional_ownership": np.nan,
        "passive_ownership": np.nan,
    }
    for metric, prefix in (("diluted_eps", "eps"), ("revenue", "revenue"),
                           ("gross_margin", "gross_margin"),
                           ("free_cash_flow", "free_cash_flow"),
                           ("bookings", "bookings")):
        estimate = con.execute("""
            SELECT estimate_value,analyst_count,forecast_dispersion,revision_breadth
            FROM earnings_consensus_snapshots
            WHERE earnings_event_id=? AND metric=? AND is_point_in_time=TRUE
              AND is_final_revised=FALSE AND estimate_as_of<?
            ORDER BY estimate_as_of DESC LIMIT 1
        """, [event_id, metric, decision_time]).fetchone()
        actual = con.execute("""
            SELECT actual_value FROM earnings_actuals
            WHERE earnings_event_id=? AND metric=? AND public_time<=?
            ORDER BY public_time DESC LIMIT 1
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
            result["has_point_in_time_consensus"] = 1.0
    for metric, output in (("guidance_eps", "guidance_eps_surprise"),
                           ("guidance_revenue", "guidance_revenue_surprise")):
        guidance = con.execute("""
            SELECT (lower_value+upper_value)/2,guidance_status FROM earnings_guidance_snapshots
            WHERE earnings_event_id=? AND metric=? AND guidance_role='new'
              AND public_time<=? AND guidance_status='AVAILABLE'
            ORDER BY public_time DESC LIMIT 1
        """, [event_id, metric, decision_time]).fetchone()
        previous = con.execute("""
            SELECT (lower_value+upper_value)/2 FROM earnings_guidance_snapshots
            WHERE earnings_event_id=? AND metric=? AND guidance_role='previous'
              AND guidance_status='AVAILABLE' AND public_time<?
            ORDER BY public_time DESC LIMIT 1
        """, [event_id, metric, event_time]).fetchone()
        if guidance and guidance[0] is not None and previous and previous[0] is not None:
            scale = max(abs(float(previous[0])), 0.01)
            result[output] = (float(guidance[0]) - float(previous[0])) / scale
            result["guidance_surprise_raw"] = result[output]
            result["has_guidance"] = 1.0
            result["guidance_status"] = "AVAILABLE"
        elif con.execute("""
            SELECT 1 FROM earnings_guidance_snapshots
            WHERE earnings_event_id=? AND metric=? AND guidance_status='NO_GUIDANCE'
            LIMIT 1
        """, [event_id, metric]).fetchone():
            result["guidance_status"] = "NO_GUIDANCE"
    options = con.execute("""
        SELECT implied_move,implied_volatility FROM earnings_options_expectations
        WHERE earnings_event_id=? AND observed_at<? ORDER BY observed_at DESC LIMIT 1
    """, [event_id, event_time]).fetchone()
    if options:
        result["implied_move"], result["implied_volatility"] = options
    positioning = con.execute("""
        SELECT days_to_cover,institutional_ownership,passive_ownership
        FROM earnings_positioning_snapshots
        WHERE earnings_event_id=? AND observed_at<? ORDER BY observed_at DESC LIMIT 1
    """, [event_id, event_time]).fetchone()
    if positioning:
        (result["days_to_cover"], result["institutional_ownership"],
         result["passive_ownership"]) = positioning
    return result


def _window_features(con, row) -> dict | None:
    bars = _read_frame(row.bars_path)
    benchmark_bars = _read_frame(row.benchmark_bars_path)
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
    if benchmark_entry is None or benchmark_delayed is None:
        return None
    exit_values = _five_day_exit(bars, benchmark_bars, session_date)
    if exit_values is None:
        return None
    exit_time = pd.Timestamp(f"{exit_values['date']} 16:00",
                             tz="America/New_York").tz_convert("UTC")

    context = _daily_context(con, row.ticker, row.benchmark_ticker, session_date)
    session_open = float(session.iloc[0]["open"])
    session_ts = _to_utc(session["ts"])
    anchor_index = int(np.searchsorted(
        session_ts.to_numpy(dtype="datetime64[ns]").astype("int64"),
                                       anchor.value, side="left"))
    event_open = float(session.iloc[anchor_index]["open"]) if anchor_index < len(session) else None
    p1 = _known_bar_close(session, anchor + pd.Timedelta(minutes=1))
    p5 = _known_bar_close(session, decision)
    p30 = _known_bar_close(session, decision)
    prior_close = _prior_rth_close(bars, session_date)
    gap = session_open / prior_close - 1 if prior_close else np.nan
    reaction1 = p1 / event_open - 1 if p1 and event_open else np.nan
    reaction5_price = _known_bar_close(session, anchor + pd.Timedelta(minutes=5))
    reaction5 = reaction5_price / event_open - 1 if reaction5_price and event_open else np.nan
    reaction30 = p30 / event_open - 1 if p30 and event_open else np.nan
    first5 = session[(session_ts >= anchor) & (session_ts < decision)]
    expected_minute_volume = ((context["trailing_adv"] / prior_close / 390)
                              if context["trailing_adv"] and prior_close else None)
    volume_ratio = (float(first5["volume"].sum()) / (expected_minute_volume * 5)
                    if expected_minute_volume and not first5.empty else np.nan)

    raw_5d = exit_values["asset_close"] / entry["price"] - 1
    benchmark_5d = exit_values["benchmark_close"] / benchmark_entry["price"] - 1
    sector_residual_5d = raw_5d - benchmark_5d
    beta = context["beta"]
    actually_hedged_5d = (raw_5d - beta * benchmark_5d
                          if beta is not None else np.nan)
    delayed_raw_5d = exit_values["asset_close"] / delayed_entry["price"] - 1
    delayed_benchmark_5d = (exit_values["benchmark_close"] /
                            benchmark_delayed["price"] - 1)
    delayed_sector_residual_5d = delayed_raw_5d - delayed_benchmark_5d
    delayed_actually_hedged_5d = (delayed_raw_5d - beta * delayed_benchmark_5d
                                  if beta is not None else np.nan)
    consensus = _consensus_actuals(con, row.earnings_event_id,
                                   row.earliest_public_time,
                                   decision.to_pydatetime())
    # A validation event whose five-day exit reaches the final period would leak
    # final-time prices into model selection. Keep it out of both cells.
    if exit_values["date"] >= FINAL_TEST_START:
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
        "gap": gap, "reaction_1m": reaction1, "reaction_5m": reaction5,
        "reaction_30m": reaction30,
        "pre_event_volatility": context["pre_event_volatility"],
        "trailing_adv": context["trailing_adv"],
        "first5_volume_ratio": volume_ratio, "beta": beta,
        "target_raw_5d": raw_5d,
        "target_sector_residual_5d": sector_residual_5d,
        "target_actually_hedged_5d": actually_hedged_5d,
        "delayed_target_raw_5d": delayed_raw_5d,
        "delayed_target_sector_residual_5d": delayed_sector_residual_5d,
        "delayed_target_actually_hedged_5d": delayed_actually_hedged_5d,
        "entry_price": entry["price"], "exit_price": exit_values["asset_close"],
        "delayed_entry_price": delayed_entry["price"],
        "benchmark_entry_price": benchmark_entry["price"],
        "benchmark_delayed_entry_price": benchmark_delayed["price"],
        "benchmark_exit_price": exit_values["benchmark_close"],
        "execution_mode": "NEXT_MINUTE_BAR_PLUS_ASSUMED_COST",
        "assumed_cost_bps_per_side": BAR_COST_BPS_PER_SIDE,
        "quote_complete": False, "quote_coverage": row.quote_coverage,
    }
    record.update(consensus)
    surprise_values = [record[name] for name in ("eps_surprise", "revenue_surprise")
                       if np.isfinite(record[name])]
    record["financial_surprise_composite"] = (float(np.mean(surprise_values))
                                               if surprise_values else np.nan)
    record["surprise_reaction_mismatch"] = (
        record["financial_surprise_composite"] - reaction30
        if np.isfinite(record["financial_surprise_composite"]) and np.isfinite(reaction30)
        else np.nan
    )
    return record


def build_feature_frame(verbose: bool = True, *,
                        before: date = FINAL_TEST_START,
                        sample_modulus: int = COARSE_SAMPLE_MODULUS,
                        sample_remainder: int = COARSE_SAMPLE_REMAINDER) -> pd.DataFrame:
    con = connect(read_only=True)
    windows = con.execute("""
        SELECT e.earnings_event_id,e.ticker,e.earliest_public_time,e.release_session,
               e.timestamp_status,e.fiscal_quarter,w.bars_path,w.quotes_path,
               w.benchmark_ticker,w.benchmark_bars_path,w.benchmark_quotes_path,
               w.quote_coverage,COALESCE(s.sector,'Unknown') sector,u.company_bucket
        FROM earnings_events e
        JOIN earnings_market_windows w USING(earnings_event_id)
        JOIN earnings_universe_snapshots u ON u.ticker=e.ticker AND u.universe_version=?
        LEFT JOIN ticker_sectors s ON s.ticker=e.ticker
        WHERE e.timestamp_status IN ('VERIFIED_EARLIEST','CONSERVATIVE_SEC_ONLY')
          AND w.status IN ('COMPLETE_QUOTES','BARS_ONLY')
          AND e.earliest_public_time < ?
        ORDER BY e.earliest_public_time,e.ticker
    """, ["earnings-alpha-v1-2021-06-30", before]).df()
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
    if not frame.empty:
        parquet_con = duckdb.connect(":memory:")
        try:
            parquet_con.register("_earnings_features", frame)
            FEATURE_PATH.unlink(missing_ok=True)
            parquet_con.execute(
                f"COPY _earnings_features TO '{FEATURE_PATH}' (FORMAT PARQUET)"
            )
        finally:
            parquet_con.close()
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
                row.target_actually_hedged_5d, now,
                json.dumps({"entry": "next one-minute bar after 30-minute decision",
                            "exit": "fifth subsequent common market close",
                            "cost_bps_per_side": BAR_COST_BPS_PER_SIDE}))
               for row in frame.itertuples(index=False)])
        write_con.close()
    return frame


def _load_feature_frame() -> pd.DataFrame:
    if not FEATURE_PATH.exists():
        return pd.DataFrame()
    con = duckdb.connect(":memory:")
    try:
        return con.execute("SELECT * FROM read_parquet(?)", [str(FEATURE_PATH)]).df()
    finally:
        con.close()


def descriptive_tables(frame: pd.DataFrame) -> dict:
    data = frame.dropna(subset=["target_sector_residual_5d"]).copy()
    if data.empty:
        return {}
    data["initial_direction"] = np.sign(data["reaction_30m"])
    data["continuation"] = data["initial_direction"] * data["target_sector_residual_5d"] > 0
    data["year"] = pd.to_datetime(data["entry_time"]).dt.year

    def table(columns):
        grouped = data.groupby(columns, observed=True, dropna=False)
        result = grouped.agg(
            n=("ticker", "size"),
            mean_raw_5d=("target_raw_5d", "mean"),
            mean_sector_residual_5d=("target_sector_residual_5d", "mean"),
            median_sector_residual_5d=("target_sector_residual_5d", "median"),
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
            mean_sector_residual_5d=("target_sector_residual_5d", "mean"),
            median_sector_residual_5d=("target_sector_residual_5d", "median"),
        ).reset_index()
        return json.loads(grouped.to_json(orient="records"))
    return {
        "reaction_30m_deciles": decile("reaction_30m"),
        "eps_surprise_deciles": decile("eps_surprise"),
        "revenue_surprise_deciles": decile("revenue_surprise"),
        "guidance_surprise_deciles": decile("guidance_revenue_surprise"),
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
    search.fit(train[numeric + categorical], train["target_sector_residual_5d"])
    return search.best_estimator_, search.best_params_


def _bar_leg(row, side: int, delayed: bool = False,
             doubled_costs: bool = False) -> float | None:
    entry = row["delayed_entry_price"] if delayed else row["entry_price"]
    benchmark_entry = (row["benchmark_delayed_entry_price"] if delayed
                       else row["benchmark_entry_price"])
    exit_price = row["exit_price"]
    benchmark_exit = row["benchmark_exit_price"]
    beta = row["beta"]
    values = [entry, exit_price, benchmark_entry, benchmark_exit, beta]
    if not all(np.isfinite(value) and value > 0 for value in values):
        return None
    asset = exit_price / entry - 1 if side > 0 else entry / exit_price - 1
    hedge_side = -side if beta >= 0 else side
    benchmark = (benchmark_exit / benchmark_entry - 1 if hedge_side > 0
                 else benchmark_entry / benchmark_exit - 1)
    round_trip_cost = BAR_COST_BPS_PER_SIDE * 2 / 1e4
    if doubled_costs:
        round_trip_cost *= 2
    return float(asset + abs(beta) * benchmark -
                 round_trip_cost * (1 + abs(beta)))


def simulate_portfolio(frame: pd.DataFrame, predictions: np.ndarray, *,
                       delayed: bool = False, doubled_costs: bool = False) -> dict:
    data = frame.copy()
    data["prediction"] = predictions
    data = data.sort_values("delayed_entry_time" if delayed else "entry_time")
    open_positions = []
    trades = []
    for row in data.to_dict(orient="records"):
        entry_time = pd.Timestamp(row["delayed_entry_time"] if delayed else row["entry_time"])
        exit_time = pd.Timestamp(row["delayed_exit_time"] if delayed else row["exit_time"])
        open_positions = [position for position in open_positions
                          if position["exit_time"] > entry_time]
        if len(open_positions) >= MAX_CONCURRENT:
            continue
        prediction = row["prediction"]
        decision = decision_from_prediction(prediction, SIGNAL_THRESHOLD_BPS)
        if decision["side"] == 0:
            continue
        side = decision["side"]
        hedge_multiplier = abs(float(row["beta"])) if np.isfinite(row["beta"]) else np.nan
        if not np.isfinite(hedge_multiplier):
            continue
        position_gross = POSITION_WEIGHT * (1 + hedge_multiplier)
        sector_gross = sum(position["asset_weight"] for position in open_positions
                           if position["sector"] == row["sector"])
        gross = sum(position["gross"] for position in open_positions)
        net = sum(position["net"] for position in open_positions)
        position_net = side * POSITION_WEIGHT * (1 - hedge_multiplier)
        if (gross + position_gross > MAX_GROSS + 1e-12 or
                sector_gross + POSITION_WEIGHT > MAX_SECTOR_GROSS + 1e-12):
            continue
        if abs(net + position_net) > MAX_NET + 1e-12:
            continue
        leg_return = _bar_leg(row, side, delayed=delayed, doubled_costs=doubled_costs)
        if leg_return is None:
            continue
        pnl = POSITION_WEIGHT * leg_return
        trade = {"earnings_event_id": row["earnings_event_id"], "ticker": row["ticker"],
                 "sector": row["sector"], "release_session": row["release_session"],
                 "entry_time": str(entry_time), "exit_time": str(exit_time),
                 "side": side, "weight": POSITION_WEIGHT, "leg_return": leg_return,
                 "gross_exposure": position_gross,
                 "pnl": pnl, "quarter": f"{entry_time.year}-Q{entry_time.quarter}"}
        trades.append(trade)
        open_positions.append({"exit_time": exit_time, "sector": row["sector"],
                               "side": side, "asset_weight": POSITION_WEIGHT,
                               "gross": position_gross, "net": position_net})
    if not trades:
        return {"n_trades": 0, "net_return": 0.0, "daily_returns": [], "trades": []}
    trade_frame = pd.DataFrame(trades)
    trade_frame["date"] = pd.to_datetime(trade_frame["exit_time"], utc=True).dt.date
    daily = trade_frame.groupby("date")["pnl"].sum()
    std = daily.std(ddof=1)
    sharpe = float(daily.mean() / std * np.sqrt(252)) if len(daily) > 1 and std > 0 else None
    return {"n_trades": len(trades), "net_return": float(trade_frame["pnl"].sum()),
            "mean_pnl_bps": float(trade_frame["pnl"].mean() * 1e4),
            "hit_rate": float((trade_frame["pnl"] > 0).mean()),
            "sharpe_annual": sharpe,
            "daily_returns": [float(value) for value in daily], "trades": trades}


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
    return {"n": len(actual), "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "mae": float(mean_absolute_error(actual, predicted))}


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
                    "target_sector_residual_5d"]] \
        .astype(str).sort_values("earnings_event_id").to_csv(index=False)
    return hashlib.sha256(values.encode()).hexdigest()[:16]


def _available_model_specs(train: pd.DataFrame) -> dict[str, tuple[list[str], list[str]]]:
    specs = {"price_only": (PRICE_NUMERIC, PRICE_CATEGORICAL)}
    consensus_coverage = float(train["has_point_in_time_consensus"].mean())
    if consensus_coverage >= 0.30:
        specs["financial_plus_reaction"] = (
            PRICE_NUMERIC + EARNINGS_NUMERIC + ["financial_surprise_composite",
                                                "surprise_reaction_mismatch"],
            PRICE_CATEGORICAL + EARNINGS_CATEGORICAL,
        )
    return specs


def _evaluate_cell(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    portfolio = simulate_portfolio(frame, prediction)
    return {
        "predictive": _metrics(frame.target_sector_residual_5d, prediction),
        "portfolio": portfolio,
        "delayed_entry": simulate_portfolio(frame, prediction, delayed=True),
        "doubled_costs": simulate_portfolio(frame, prediction, doubled_costs=True),
        "unseen_company_robustness": _metrics(
            frame.loc[frame.company_bucket == "UNSEEN_COMPANY",
                      "target_sector_residual_5d"],
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
        lock_spec: bool = False, final_test: bool = False) -> dict:
    """Run validation by default; the frozen final year requires a prior lock.

    Validation chooses the model. ``final_test=True`` is irreversible at the
    report level: once written, later calls return the existing result.
    """
    if final_test and FINAL_REPORT_PATH.exists():
        existing = json.loads(FINAL_REPORT_PATH.read_text())
        return {"status": "FINAL_TEST_ALREADY_OPENED", "result": existing}
    if final_test and not SPEC_LOCK_PATH.exists():
        return {"status": "FINAL_TEST_BLOCKED_SPEC_NOT_LOCKED"}

    data = (frame.copy() if frame is not None else
            (_load_feature_frame() if FEATURE_PATH.exists()
             else build_feature_frame(verbose=verbose)))
    if data.empty:
        result = {"status": "NOT_EVALUATED_DATA_INCOMPLETE",
                  "reason": "no Tier 1/2 earnings event bar windows"}
        REPORT_PATH.write_text(json.dumps(result, indent=2))
        return result
    data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True)
    usable = data.dropna(subset=["target_sector_residual_5d"]) \
        .sort_values("entry_time").copy()
    train = usable[usable.time_bucket == "TRAIN_TIME"].copy()
    validation = usable[usable.time_bucket == "VALIDATION_TIME"].copy()
    final = usable[usable.time_bucket == "FINAL_TEST_TIME"].copy()
    prefinal = usable[usable.time_bucket != "FINAL_TEST_TIME"].copy()
    counts = {"features": len(data), "usable": len(usable), "train": len(train),
              "validation": len(validation), "final_frozen": len(final),
              "unseen_company_validation": int(
                  (validation.company_bucket == "UNSEEN_COMPANY").sum())}
    if len(train) < MIN_TRAIN or len(validation) < MIN_VALIDATION:
        result = {
            "status": "NOT_EVALUATED_DATA_INCOMPLETE", "counts": counts,
            "required": {"train": MIN_TRAIN, "validation": MIN_VALIDATION},
            "descriptive": descriptive_tables(prefinal),
            "final_test_outcomes_evaluated": False,
        }
        REPORT_PATH.write_text(json.dumps(result, indent=2, default=str))
        return result

    dataset_hash = _dataset_hash(usable if final_test else prefinal)
    splits = {"train_end": str(VALIDATION_START - timedelta(days=1)),
              "validation_start": str(VALIDATION_START),
              "validation_end": str(FINAL_TEST_START - timedelta(days=1)),
              "final_test_start": str(FINAL_TEST_START),
              "primary_test": "future time across all eligible companies",
              "unseen_company": "additional robustness only"}

    if final_test:
        if final.empty:
            return {"status": "FINAL_TEST_BLOCKED_DATA_INCOMPLETE",
                    "counts": counts}
        spec = json.loads(SPEC_LOCK_PATH.read_text())
        model_name = spec["model_name"]
        numeric, categorical = spec["numeric"], spec["categorical"]
        model = _pipeline(numeric, categorical)
        model.set_params(**spec["elastic_net_params"])
        model.fit(prefinal[numeric + categorical],
                  prefinal["target_sector_residual_5d"])
        prediction = model.predict(final[numeric + categorical])
        evaluated = _evaluate_cell(final, prediction)
        stability = _stability_and_concentration(evaluated["portfolio"])
        dsr = _deflated_sharpe(evaluated["portfolio"]["daily_returns"],
                               int(spec["experiment_trials"]))
        result = {
            "status": "FINAL_TEST_OPENED", "model_name": model_name,
            "counts": counts, "splits": splits, "dataset_hash": dataset_hash,
            "experiment_id": hashlib.sha256(
                f"{SPRINT_VERSION}|{dataset_hash}|FINAL|{model_name}".encode()
            ).hexdigest()[:20],
            "result": evaluated, "stability": stability,
            "deflated_sharpe": dsr, "spec_lock": spec,
            "final_test_outcomes_evaluated": True,
        }
        FINAL_REPORT_PATH.write_text(json.dumps(result, indent=2, default=str))
        _persist_experiment(result, result["status"])
        return result

    model_specs = _available_model_specs(train)
    models, params, validation_results, predictions = {}, {}, {}, {}
    for model_name, (numeric, categorical) in model_specs.items():
        models[model_name], params[model_name] = _fit(train, numeric, categorical)
        predictions[model_name] = models[model_name].predict(
            validation[numeric + categorical]
        )
        validation_results[model_name] = _evaluate_cell(
            validation, predictions[model_name]
        )
    selected = min(model_specs, key=lambda name:
                   validation_results[name]["predictive"]["rmse"])
    selected_result = validation_results[selected]
    naive_rmse = float(np.sqrt(np.mean(
        np.square(validation["target_sector_residual_5d"].to_numpy())
    )))
    screening_gates = {
        "predictive_lift_vs_zero": selected_result["predictive"]["rmse"] < naive_rmse,
        "positive_net_after_15bps_per_side":
            selected_result["portfolio"]["net_return"] > 0,
        "positive_with_60min_delayed_entry":
            selected_result["delayed_entry"]["net_return"] > 0,
        "positive_with_30bps_per_side":
            selected_result["doubled_costs"]["net_return"] > 0,
    }
    structured_tested = "financial_plus_reaction" in model_specs
    passed = all(screening_gates.values())
    experiment_id = hashlib.sha256(
        f"{SPRINT_VERSION}|{dataset_hash}|VALIDATION|{selected}|{_git_hash()}".encode()
    ).hexdigest()[:20]
    result = {
        "status": "VALIDATION_BAR_SCREEN_PASSED" if passed
                  else "VALIDATION_BAR_SCREEN_REJECTED",
        "experiment_id": experiment_id, "dataset_hash": dataset_hash,
        "counts": counts, "splits": splits, "selected_model": selected,
        "models": validation_results, "elastic_net_params": params,
        "screening_gates": screening_gates,
        "structured_surprise_tested": structured_tested,
        "structured_surprise_blocker": None if structured_tested else
            "point-in-time consensus coverage below 30%",
        "catboost_allowed": bool(passed and structured_tested and
                                 selected == "financial_plus_reaction"),
        "descriptive": descriptive_tables(prefinal),
        "final_test_outcomes_evaluated": False,
    }
    if lock_spec:
        numeric, categorical = model_specs[selected]
        con = connect(read_only=True)
        trials = max(32, 32 + con.execute(
            "SELECT COUNT(*) FROM earnings_experiments"
        ).fetchone()[0])
        con.close()
        lock = {
            "locked_at": datetime.now(timezone.utc).isoformat(),
            "validation_experiment_id": experiment_id,
            "model_name": selected, "numeric": numeric,
            "categorical": categorical,
            "elastic_net_params": params[selected],
            "signal_threshold_bps": SIGNAL_THRESHOLD_BPS,
            "bar_cost_bps_per_side": BAR_COST_BPS_PER_SIDE,
            "experiment_trials": trials, "splits": splits,
        }
        if SPEC_LOCK_PATH.exists():
            raise EarningsStudyError("model specification is already locked")
        SPEC_LOCK_PATH.write_text(json.dumps(lock, indent=2, default=str))
        result["specification_locked"] = True
    _persist_experiment(result, result["status"])
    REPORT_PATH.write_text(json.dumps(result, indent=2, default=str))
    if verbose:
        print(f"Earnings five-day validation: {result['status']} "
              f"selected={selected} gates={screening_gates}")
    return result
