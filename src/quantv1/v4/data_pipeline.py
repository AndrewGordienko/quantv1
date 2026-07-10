"""Reliable, resumable minute-bar ingestion (fixes review point #5).

Two clean phases so we never do millions of page-by-page conflict checks against
a single locked DuckDB file:

  1. FETCH  — pull each ticker from Polygon and write ONE Parquet file per ticker.
              Pure network + local file I/O; no DB lock held. Resumable: a ticker
              whose watermark already covers the requested range is skipped.
  2. LOAD   — a single bulk `read_parquet(...)` insert rebuilds `bars_minute` in
              seconds. Parquet is the source of truth.

`completeness()` audits expected vs actual trading days per ticker so we can see
gaps (the NVDA-only-26k problem) instead of silently trusting "some rows exist".
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl

from ..config import DATA_DIR
from ..db import connect
from ..ingest.polygon_data import _key, _get, _BARS

PARQUET = DATA_DIR / "parquet" / "bars"
PARQUET.mkdir(parents=True, exist_ok=True)
SOURCE = "polygon_minute"
_COLS = ["ticker", "ts", "open", "high", "low", "close", "volume", "trades", "vwap"]


def _watermark(con, ticker) -> dict | None:
    r = con.execute("SELECT req_start, req_end, row_count, status FROM ingest_watermarks "
                    "WHERE ticker=? AND source=?", [ticker, SOURCE]).fetchone()
    if not r:
        return None
    return {"req_start": str(r[0]), "req_end": str(r[1]), "row_count": r[2], "status": r[3]}


def _fetch_ticker(ticker: str, start: str, end: str, key: str) -> pl.DataFrame | None:
    url = (_BARS.format(tk=ticker, a=start, b=end) +
           f"?adjusted=true&sort=asc&limit=50000&apiKey={key}")
    rows = []
    while url:
        data = _get(url)
        if not data:
            break
        for b in data.get("results", []):
            rows.append((ticker, b["t"], b.get("o"), b.get("h"), b.get("l"),
                         b.get("c"), b.get("v"), b.get("n"), b.get("vw")))
        nxt = data.get("next_url")
        url = f"{nxt}&apiKey={key}" if nxt else None
    if not rows:
        return None
    df = pl.DataFrame(rows, schema=["ticker", "t_ms", "open", "high", "low", "close",
                                    "volume", "trades", "vwap"], orient="row")
    # ms epoch (UTC) -> tz-naive UTC datetime to match the bars_minute convention
    df = df.with_columns(
        pl.from_epoch(pl.col("t_ms"), time_unit="ms").dt.replace_time_zone(None).alias("ts")
    ).drop("t_ms").select(_COLS)
    return df


def fetch(tickers: list[str], start: str, end: str, force: bool = False,
          verbose: bool = True) -> dict:
    """FETCH phase: Polygon -> one Parquet per ticker. Resumable via watermarks."""
    key = _key()
    if not key:
        from ..ingest.polygon_data import _setup
        _setup()
        return {"error": "no key"}
    con = connect()
    done, skipped, failed = 0, 0, 0
    for tk in tickers:
        wm = _watermark(con, tk)
        if wm and not force and wm["status"] == "complete" \
                and wm["req_start"] <= start and wm["req_end"] >= end:
            skipped += 1
            if verbose:
                print(f"  {tk}: watermark covers range ({wm['row_count']} rows), skip")
            continue
        try:
            df = _fetch_ticker(tk, start, end, key)
        except Exception as e:  # noqa: BLE001
            failed += 1
            if verbose:
                print(f"  {tk}: fetch failed: {e}")
            continue
        if df is None or df.is_empty():
            failed += 1
            continue
        df.write_parquet(PARQUET / f"{tk}.parquet")
        tdays = df.select(pl.col("ts").dt.date().n_unique()).item()
        con.execute("INSERT OR REPLACE INTO ingest_watermarks VALUES (?,?,?,?,?,?,?,?,?)",
                    [tk, SOURCE, start, end, df.select(pl.col("ts").max()).item(),
                     df.height, tdays, "complete", datetime.now(timezone.utc)])
        done += 1
        if verbose:
            print(f"  {tk}: {df.height} rows, {tdays} trading days -> parquet")
    con.close()
    if verbose:
        print(f"FETCH: {done} written, {skipped} skipped, {failed} failed")
    return {"written": done, "skipped": skipped, "failed": failed}


def load(verbose: bool = True) -> dict:
    """LOAD phase: single bulk Parquet -> bars_minute (rebuild). Fast."""
    files = sorted(PARQUET.glob("*.parquet"))
    if not files:
        return {"error": "no parquet staged — run fetch() first"}
    con = connect()
    con.execute("DELETE FROM bars_minute")
    con.execute(f"""
        INSERT INTO bars_minute
        SELECT ticker, ts, open, high, low, close, volume, trades, vwap
        FROM read_parquet('{PARQUET}/*.parquet')
    """)
    n = con.execute("SELECT COUNT(*) FROM bars_minute").fetchone()[0]
    tk = con.execute("SELECT COUNT(DISTINCT ticker) FROM bars_minute").fetchone()[0]
    con.close()
    if verbose:
        print(f"LOAD: {n} bars across {tk} tickers from {len(files)} parquet files")
    return {"rows": n, "tickers": tk, "files": len(files)}


def completeness(verbose: bool = True) -> list[dict]:
    """Expected vs actual trading days per ticker (surfaces gaps/incompleteness)."""
    con = connect(read_only=True)
    wms = con.execute("SELECT ticker, req_start, req_end, row_count, trading_days "
                      "FROM ingest_watermarks WHERE source=?", [SOURCE]).fetchall()
    rep = []
    for tk, s, e, rc, td in wms:
        exp = int(np.busday_count(str(s), str(e)))     # rough expected trading days
        rep.append({"ticker": tk, "start": str(s), "end": str(e), "rows": rc,
                    "trading_days": td, "expected_days": exp,
                    "coverage": round(td / exp, 3) if exp else None,
                    "complete": bool(td >= exp * 0.95)})
    con.close()
    if verbose:
        print("=== completeness (trading days actual/expected) ===")
        for r in sorted(rep, key=lambda x: x["coverage"] or 0):
            flag = "OK " if r["complete"] else "GAP"
            print(f"  [{flag}] {r['ticker']:6s} {r['trading_days']}/{r['expected_days']} days "
                  f"({r['coverage']}) {r['rows']} rows")
    return rep


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        load(); completeness()
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        completeness()
    else:
        print("usage: fetch via a driver; then `load` then `report`")
