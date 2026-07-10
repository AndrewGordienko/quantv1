"""Resumable event-window bars and historical NBBO quote acquisition.

Unlike the general minute store, this fetches only the window needed around each
earnings release. Quote absence is explicit: a bars-only window can feed
descriptive tables but can never clear an executable promotion gate.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as wall_time, timedelta, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import polars as pl

from ..config import DATA_DIR
from ..db import connect
from ..ingest.polygon_data import _get, _key
from .data_pipeline import _fetch_ticker

WINDOW_ROOT = DATA_DIR / "parquet" / "earnings_windows"
BENCHMARK_ROOT = DATA_DIR / "parquet" / "earnings_benchmarks"
_QUOTES = "https://api.polygon.io/v3/quotes/{ticker}"
_ET = ZoneInfo("America/New_York")
SECTOR_ETF = {
    "communication services": "XLC", "consumer discretionary": "XLY",
    "consumer staples": "XLP", "energy": "XLE", "financials": "XLF",
    "health care": "XLV", "industrials": "XLI",
    "information technology": "XLK", "materials": "XLB",
    "real estate": "XLRE", "utilities": "XLU", "technology": "XLK",
    "financial services": "XLF",
    "consumer cyclical": "XLY", "consumer defensive": "XLP",
    "healthcare": "XLV", "basic materials": "XLB",
}


def _next_weekday(value):
    result = value + timedelta(days=1)
    while result.weekday() >= 5:
        result += timedelta(days=1)
    return result


def window_bounds(public_time: datetime, release_session: str) -> tuple[datetime, datetime]:
    """Bounds for pre-event context through the fifth subsequent market close."""
    aware = public_time.replace(tzinfo=timezone.utc)
    local = aware.astimezone(_ET)
    # Use one vendor/adjustment basis for prior close, entry and 5-day exit.
    start = aware - timedelta(days=4)
    session_date = local.date()
    if local.weekday() >= 5:
        session_date = _next_weekday(session_date)
    elif release_session == "AMC":
        session_date = _next_weekday(session_date)
    elif release_session == "UNKNOWN" and local.hour >= 16:
        session_date = _next_weekday(session_date)
    # Holiday buffer for five *subsequent* regular sessions.
    end_local = datetime.combine(session_date + timedelta(days=10),
                                 wall_time(16, 15), tzinfo=_ET)
    end = end_local.astimezone(timezone.utc)
    if end <= aware:
        end = aware + timedelta(days=2)
    return start.replace(tzinfo=None), end.replace(tzinfo=None)


def _fetch_quotes(ticker: str, start: datetime, end: datetime, key: str) -> pl.DataFrame:
    params = {
        "timestamp.gte": start.replace(tzinfo=timezone.utc).isoformat(),
        "timestamp.lte": end.replace(tzinfo=timezone.utc).isoformat(),
        "sort": "timestamp", "order": "asc", "limit": 50000, "apiKey": key,
    }
    url = _QUOTES.format(ticker=ticker) + "?" + urlencode(params)
    rows = []
    while url:
        payload = _get(url)
        if not payload:
            break
        for quote in payload.get("results", []):
            timestamp = quote.get("sip_timestamp")
            if timestamp is None:
                continue
            rows.append((
                ticker,
                datetime.fromtimestamp(timestamp / 1e9, tz=timezone.utc).replace(tzinfo=None),
                quote.get("bid_price"), quote.get("ask_price"),
                quote.get("bid_size"), quote.get("ask_size"),
                quote.get("bid_exchange"), quote.get("ask_exchange"),
                quote.get("sequence_number"),
            ))
        next_url = payload.get("next_url")
        url = f"{next_url}&apiKey={key}" if next_url else None
    if not rows:
        return pl.DataFrame(schema={
            "ticker": pl.String, "ts": pl.Datetime("us"), "bid": pl.Float64,
            "ask": pl.Float64, "bid_size": pl.Float64, "ask_size": pl.Float64,
            "bid_exchange": pl.Int64, "ask_exchange": pl.Int64,
            "sequence_number": pl.Int64,
        })
    return pl.DataFrame(rows, schema=[
        "ticker", "ts", "bid", "ask", "bid_size", "ask_size",
        "bid_exchange", "ask_exchange", "sequence_number",
    ], orient="row")


def _coverage(bars: pl.DataFrame, quotes: pl.DataFrame) -> float:
    if bars.is_empty() or quotes.is_empty():
        return 0.0
    bar_minutes = bars.select(pl.col("ts").dt.truncate("1m").n_unique()).item()
    quote_minutes = quotes.select(pl.col("ts").dt.truncate("1m").n_unique()).item()
    return float(min(quote_minutes / max(bar_minutes, 1), 1.0))


def _ensure_benchmark_histories(benchmarks: set[str], start: datetime, end: datetime,
                                key: str) -> dict[str, Path]:
    """Fetch each sector ETF once, rather than once per earnings event."""
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {benchmark: BENCHMARK_ROOT / f"{benchmark}_{start.date()}_{end.date()}.parquet"
             for benchmark in benchmarks}

    def fetch(benchmark: str) -> tuple[str, pl.DataFrame | None]:
        path = paths[benchmark]
        if path.exists():
            return benchmark, None
        return benchmark, _fetch_ticker(benchmark, str(start.date()), str(end.date()), key)

    # This is bounded well below vendor rate limits and avoids N_event duplicate
    # requests for the same sector ETF.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fetch, benchmark) for benchmark in sorted(benchmarks)]
        for future in as_completed(futures):
            benchmark, bars = future.result()
            if bars is None:
                continue
            if bars.is_empty():
                raise RuntimeError(f"no benchmark bars for {benchmark}")
            bars.write_parquet(paths[benchmark])
    return paths


def fetch_windows(*, include_conservative: bool = False, max_events: int | None = None,
                  force: bool = False, with_quotes: bool = False,
                  before: datetime | None = None,
                  sample_modulus: int | None = None, sample_remainder: int = 0,
                  verbose: bool = True) -> dict:
    key = _key()
    if not key:
        return {"error": "no POLYGON_API_KEY"}
    con = connect()
    where = ("WHERE e.timestamp_status IN ('VERIFIED_EARLIEST','CONSERVATIVE_SEC_ONLY')"
             if include_conservative else
             "WHERE e.timestamp_status='VERIFIED_EARLIEST'")
    limit = f"LIMIT {int(max_events)}" if max_events else ""
    before_clause = "AND e.earliest_public_time < ?" if before else ""
    events = con.execute(f"""
        SELECT e.earnings_event_id,e.ticker,e.earliest_public_time,e.release_session,
               w.status existing_status,lower(s.sector) sector
        FROM earnings_events e LEFT JOIN earnings_market_windows w USING(earnings_event_id)
        LEFT JOIN ticker_sectors s ON s.ticker=e.ticker
        {where}
        {before_clause}
        ORDER BY e.earliest_public_time,e.ticker {limit}
    """, [before] if before else []).fetchall()
    if sample_modulus:
        if sample_modulus < 2 or not 0 <= sample_remainder < sample_modulus:
            raise ValueError("invalid deterministic screening sample")
        events = [row for row in events if
                  int(hashlib.sha256(row[0].encode()).hexdigest()[:8], 16) %
                  sample_modulus == sample_remainder]
    if not events:
        con.close()
        return {"events": 0, "complete_quotes": 0, "bars_only": 0,
                "failed": 0, "skipped": 0, "quote_requests_enabled": with_quotes}
    bounds = [window_bounds(public_time, release_session)
              for _, _, public_time, release_session, _, _ in events]
    benchmark_paths = _ensure_benchmark_histories(
        {SECTOR_ETF.get(sector or "", "SPY") for *_, sector in events},
        min(start for start, _ in bounds), max(end for _, end in bounds), key,
    )
    complete = bars_only = failed = skipped = 0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for index, (event_id, ticker, public_time, release_session, existing, sector) in enumerate(events, 1):
        if existing in {"COMPLETE_QUOTES", "BARS_ONLY"} and not force:
            skipped += 1
            continue
        start, end = window_bounds(public_time, release_session)
        directory = WINDOW_ROOT / event_id
        directory.mkdir(parents=True, exist_ok=True)
        bars_path = directory / "bars.parquet"
        quotes_path = directory / "quotes.parquet"
        benchmark = SECTOR_ETF.get(sector or "", "SPY")
        benchmark_bars_path = benchmark_paths[benchmark]
        benchmark_quotes_path = None
        try:
            bars = _fetch_ticker(ticker, str(start.date()), str(end.date()), key)
            if bars is None:
                raise RuntimeError("no minute bars")
            bars = bars.filter((pl.col("ts") >= start) & (pl.col("ts") <= end))
            bars.write_parquet(bars_path)
            quotes = (_fetch_quotes(ticker, start, end, key) if with_quotes else
                      pl.DataFrame())
            if not quotes.is_empty():
                quotes.write_parquet(quotes_path)
            if with_quotes:
                benchmark_bars = pl.read_parquet(benchmark_bars_path)
                benchmark_directory = WINDOW_ROOT / "_benchmark_quotes" / benchmark / \
                    f"{start.date()}_{end.date()}"
                benchmark_directory.mkdir(parents=True, exist_ok=True)
                benchmark_quotes_path = benchmark_directory / "quotes.parquet"
                if benchmark_quotes_path.exists():
                    benchmark_quotes = pl.read_parquet(benchmark_quotes_path)
                else:
                    benchmark_quotes = _fetch_quotes(benchmark, start, end, key)
                    if not benchmark_quotes.is_empty():
                        benchmark_quotes.write_parquet(benchmark_quotes_path)
            else:
                benchmark_quotes = pl.DataFrame()
            if with_quotes and not quotes.is_empty() and not benchmark_quotes.is_empty():
                status = "COMPLETE_QUOTES"
                complete += 1
            else:
                status = "BARS_ONLY"
                bars_only += 1
            coverage = (min(_coverage(bars, quotes),
                            _coverage(benchmark_bars, benchmark_quotes))
                        if with_quotes else 0.0)
            con.execute("""
                INSERT INTO earnings_market_windows
                    (earnings_event_id,ticker,window_start,window_end,bars_source,
                     quotes_source,bars_path,quotes_path,benchmark_ticker,
                     benchmark_bars_path,benchmark_quotes_path,bar_rows,quote_rows,
                     quote_coverage,status,retrieved_at,metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (earnings_event_id) DO UPDATE SET
                    window_start=excluded.window_start,window_end=excluded.window_end,
                    bars_source=excluded.bars_source,quotes_source=excluded.quotes_source,
                    bars_path=excluded.bars_path,quotes_path=excluded.quotes_path,
                    benchmark_ticker=excluded.benchmark_ticker,
                    benchmark_bars_path=excluded.benchmark_bars_path,
                    benchmark_quotes_path=excluded.benchmark_quotes_path,
                    bar_rows=excluded.bar_rows,quote_rows=excluded.quote_rows,
                    quote_coverage=excluded.quote_coverage,status=excluded.status,
                    retrieved_at=excluded.retrieved_at,metadata=excluded.metadata
            """, [event_id, ticker, start, end, "polygon_minute", "polygon_nbbo",
                  str(bars_path), str(quotes_path) if not quotes.is_empty() else None,
                  benchmark, str(benchmark_bars_path),
                  (str(benchmark_quotes_path) if benchmark_quotes_path and
                   not benchmark_quotes.is_empty() else None),
                  bars.height, quotes.height, coverage, status, now,
                  json.dumps({"event_window_only": True,
                              "coarse_bar_screen_eligible": True,
                              "quotes_required_for_final_promotion_only": True})])
        except Exception as exc:  # noqa: BLE001
            failed += 1
            con.execute("""
                INSERT INTO earnings_market_windows
                    (earnings_event_id,ticker,window_start,window_end,status,retrieved_at,metadata)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT (earnings_event_id) DO UPDATE SET
                    status=excluded.status,retrieved_at=excluded.retrieved_at,
                    metadata=excluded.metadata
            """, [event_id, ticker, start, end, "FAILED", now,
                  json.dumps({"error": str(exc)[:500]})])
        if verbose and index % 25 == 0:
            print(f"  windows {index}/{len(events)} quotes={complete} "
                  f"bars_only={bars_only} failed={failed}")
    con.close()
    return {"events": len(events), "complete_quotes": complete,
            "bars_only": bars_only, "failed": failed, "skipped": skipped,
            "quote_requests_enabled": with_quotes,
            "before": before.isoformat() if before else None,
            "sample_modulus": sample_modulus,
            "sample_remainder": sample_remainder if sample_modulus else None}


def fetch_bar_windows_parallel(*, include_conservative: bool = False,
                               before: datetime | None = None,
                               sample_modulus: int | None = None,
                               sample_remainder: int = 0, force: bool = False,
                               workers: int = 3, verbose: bool = True) -> dict:
    """Concurrent bar-only acquisition; DB provenance writes stay serialized."""
    if workers < 1:
        raise ValueError("workers must be positive")
    key = _key()
    if not key:
        return {"error": "no POLYGON_API_KEY"}
    con = connect()
    where = ("WHERE e.timestamp_status IN ('VERIFIED_EARLIEST','CONSERVATIVE_SEC_ONLY')"
             if include_conservative else "WHERE e.timestamp_status='VERIFIED_EARLIEST'")
    before_clause = "AND e.earliest_public_time < ?" if before else ""
    events = con.execute(f"""
        SELECT e.earnings_event_id,e.ticker,e.earliest_public_time,e.release_session,
               w.status existing_status,lower(s.sector) sector
        FROM earnings_events e LEFT JOIN earnings_market_windows w USING(earnings_event_id)
        LEFT JOIN ticker_sectors s ON s.ticker=e.ticker
        {where} {before_clause}
        ORDER BY e.earliest_public_time,e.ticker
    """, [before] if before else []).fetchall()
    if sample_modulus:
        events = [row for row in events if
                  int(hashlib.sha256(row[0].encode()).hexdigest()[:8], 16) %
                  sample_modulus == sample_remainder]
    if not events:
        con.close()
        return {"events": 0, "bars_only": 0, "failed": 0, "skipped": 0}
    bounds = [window_bounds(public_time, release_session)
              for _, _, public_time, release_session, _, _ in events]
    benchmark_paths = _ensure_benchmark_histories(
        {SECTOR_ETF.get(sector or "", "SPY") for *_, sector in events},
        min(start for start, _ in bounds), max(end for _, end in bounds), key,
    )
    pending = [row for row in events if force or row[4] not in {"BARS_ONLY", "COMPLETE_QUOTES"}]
    skipped = len(events) - len(pending)

    def fetch(row):
        event_id, ticker, public_time, release_session, _, sector = row
        start, end = window_bounds(public_time, release_session)
        directory = WINDOW_ROOT / event_id
        directory.mkdir(parents=True, exist_ok=True)
        bars_path = directory / "bars.parquet"
        try:
            bars = _fetch_ticker(ticker, str(start.date()), str(end.date()), key)
            if bars is None:
                raise RuntimeError("no minute bars")
            bars = bars.filter((pl.col("ts") >= start) & (pl.col("ts") <= end))
            if bars.is_empty():
                raise RuntimeError("no bars in event window")
            bars.write_parquet(bars_path)
            benchmark = SECTOR_ETF.get(sector or "", "SPY")
            return {"row": row, "start": start, "end": end, "bars_path": str(bars_path),
                    "bar_rows": bars.height, "benchmark": benchmark,
                    "benchmark_path": str(benchmark_paths[benchmark]), "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"row": row, "start": start, "end": end, "error": str(exc)[:500]}

    bars_only = failed = 0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, row) for row in pending]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            event_id, ticker, *_ = result["row"]
            if result["error"] is None:
                con.execute("""
                    INSERT INTO earnings_market_windows
                        (earnings_event_id,ticker,window_start,window_end,bars_source,
                         quotes_source,bars_path,quotes_path,benchmark_ticker,
                         benchmark_bars_path,benchmark_quotes_path,bar_rows,quote_rows,
                         quote_coverage,status,retrieved_at,metadata)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT (earnings_event_id) DO UPDATE SET
                        window_start=excluded.window_start,window_end=excluded.window_end,
                        bars_source=excluded.bars_source,quotes_source=excluded.quotes_source,
                        bars_path=excluded.bars_path,quotes_path=NULL,
                        benchmark_ticker=excluded.benchmark_ticker,
                        benchmark_bars_path=excluded.benchmark_bars_path,
                        benchmark_quotes_path=NULL,bar_rows=excluded.bar_rows,quote_rows=0,
                        quote_coverage=0,status='BARS_ONLY',retrieved_at=excluded.retrieved_at,
                        metadata=excluded.metadata
                """, [event_id, ticker, result["start"], result["end"],
                      "polygon_minute", None, result["bars_path"], None,
                      result["benchmark"], result["benchmark_path"], None,
                      result["bar_rows"], 0, 0.0, "BARS_ONLY", now,
                      json.dumps({"event_window_only": True,
                                  "coarse_bar_screen_eligible": True,
                                  "market_window_version": "earnings-5d-bars-v2",
                                  "quotes_required_for_final_promotion_only": True})])
                bars_only += 1
            else:
                failed += 1
                con.execute("""
                    INSERT INTO earnings_market_windows
                        (earnings_event_id,ticker,window_start,window_end,status,retrieved_at,metadata)
                    VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT (earnings_event_id) DO UPDATE SET
                        status='FAILED',retrieved_at=excluded.retrieved_at,
                        metadata=excluded.metadata
                """, [event_id, ticker, result["start"], result["end"], "FAILED", now,
                      json.dumps({"error": result["error"],
                                  "market_window_version": "earnings-5d-bars-v2"})])
            if verbose and index % 50 == 0:
                print(f"  parallel windows {index}/{len(pending)} bars_only={bars_only} "
                      f"failed={failed}", flush=True)
    con.close()
    return {"events": len(events), "bars_only": bars_only, "failed": failed,
            "skipped": skipped, "workers": workers,
            "sample_modulus": sample_modulus, "before": before.isoformat() if before else None}


def entitlement_preflight(ticker: str, start: datetime, end: datetime) -> dict:
    """Check bars/quote availability without exposing credentials."""
    key = _key()
    if not key:
        return {"minute_bars": False, "historical_nbbo": False,
                "promotion_blocked": True, "reason": "missing API key"}
    bars = _fetch_ticker(ticker, str(start.date()), str(end.date()), key)
    quotes = _fetch_quotes(ticker, start, end, key)
    bar_rows = 0 if bars is None else bars.height
    return {"minute_bars": bar_rows > 0, "historical_nbbo": not quotes.is_empty(),
            "bar_rows": bar_rows, "quote_rows": quotes.height,
            "promotion_blocked": quotes.is_empty(),
            "reason": None if not quotes.is_empty() else
                      "historical NBBO unavailable under current entitlement"}
