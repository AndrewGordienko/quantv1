"""Ingest daily adjusted OHLCV via yfinance into DuckDB `prices`.

Universe = every ticker that appears in `trades`, plus SPY and the sector ETFs.
Split/dividend-adjusted closes are used throughout so returns are comparable
across time. Delisted or renamed tickers simply return nothing from yfinance;
we record what we get and let downstream code treat missing prices as "not
investable" (honest survivorship handling — we never fabricate a price).
"""

from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from .. import config
from ..db import connect

START = "2012-01-01"
BATCH = 60          # tickers per yfinance download call
PAUSE = 1.0         # seconds between batches (rate-limit courtesy)


def target_universe(con) -> list[str]:
    tickers = [r[0] for r in con.execute(
        "SELECT DISTINCT ticker FROM trades WHERE ticker IS NOT NULL"
    ).fetchall()]
    extra = [config.BENCHMARK_TICKER, *config.SECTOR_ETFS]
    return sorted(set(tickers) | set(extra))


def _download_batch(tickers: list[str], start: str) -> pd.DataFrame:
    """Return a long-form frame [ticker,date,open,high,low,close,volume]."""
    df = yf.download(
        tickers, start=start, auto_adjust=True, progress=False,
        threads=True, group_by="ticker",
    )
    if df is None or df.empty:
        return pd.DataFrame()
    frames = []
    # yfinance returns a single-level frame for one ticker, MultiIndex for many.
    syms = tickers if len(tickers) > 1 else [tickers[0]]
    for sym in syms:
        try:
            sub = df[sym] if len(tickers) > 1 else df
        except KeyError:
            continue
        sub = sub.dropna(how="all")
        if sub.empty:
            continue
        sub = sub.reset_index().rename(columns={
            "Date": "date", "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        sub["ticker"] = sym
        frames.append(sub[["ticker", "date", "open", "high", "low", "close", "volume"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ingest(incremental: bool = True, verbose: bool = True) -> dict:
    con = connect()
    universe = target_universe(con)

    # Incremental: only pull from the day after the latest stored bar per ticker.
    last_dates = dict(con.execute(
        "SELECT ticker, MAX(date) FROM prices GROUP BY ticker"
    ).fetchall()) if incremental else {}

    n_rows, n_ok, n_fail = 0, 0, 0
    for i in range(0, len(universe), BATCH):
        batch = universe[i:i + BATCH]
        # Use the oldest per-ticker start in the batch; over-fetch is deduped on upsert.
        starts = [last_dates.get(t) for t in batch if last_dates.get(t)]
        start = START
        if incremental and len(starts) == len(batch) and starts:
            start = (min(starts) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            frame = _download_batch(batch, start)
        except Exception as e:  # noqa: BLE001 - yfinance raises many transient errors
            if verbose:
                print(f"  batch {i//BATCH} failed: {e}")
            n_fail += len(batch)
            continue
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"]).dt.date
            con.execute("CREATE TEMP TABLE _p AS SELECT * FROM prices WHERE 0=1")
            con.executemany(
                "INSERT INTO _p VALUES (?,?,?,?,?,?,?)",
                frame[["ticker", "date", "open", "high", "low", "close", "volume"]]
                .itertuples(index=False, name=None),
            )
            con.execute(
                "DELETE FROM prices WHERE (ticker, date) IN (SELECT ticker, date FROM _p)"
            )
            con.execute("INSERT INTO prices SELECT * FROM _p")
            con.execute("DROP TABLE _p")
            n_rows += len(frame)
            n_ok += frame["ticker"].nunique()
        if verbose:
            print(f"  batch {i//BATCH + 1}/{(len(universe)-1)//BATCH + 1} "
                  f"({len(batch)} tickers) -> {len(frame)} rows")
        time.sleep(PAUSE)

    total = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    covered = con.execute("SELECT COUNT(DISTINCT ticker) FROM prices").fetchone()[0]
    con.close()
    stats = {"universe": len(universe), "tickers_with_data": covered,
             "rows_added": n_rows, "db_total": total}
    if verbose:
        print(f"Prices: universe={len(universe)} covered={covered} "
              f"added={n_rows} total={total}")
    return stats


if __name__ == "__main__":
    ingest()
