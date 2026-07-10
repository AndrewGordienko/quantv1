"""Ingest ~2 years of hourly bars for a liquid universe (fast-trigger PoC).

Free yfinance intraday history caps at 730 days for 60-minute bars, so this is a
proof-of-concept resolution — enough to build and honestly evaluate the
fast-trigger harness, not the real minute-level system (which needs Alpaca/
Polygon/IBKR). Universe = sector ETFs + SPY/QQQ + the most-traded, sector-known
congressional tickers (a liquid cross-section for sector-relative work).
"""

from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from .. import config
from ..db import connect

N_NAMES = 120          # liquid congressional names to include
BATCH = 40


def universe(con) -> list[str]:
    names = [r[0] for r in con.execute("""
        SELECT t.ticker
        FROM trades t
        JOIN ticker_sectors s ON t.ticker = s.ticker AND s.sector <> 'Unknown'
        GROUP BY t.ticker
        ORDER BY COUNT(*) DESC
        LIMIT ?
    """, [N_NAMES]).fetchall()]
    extra = [config.BENCHMARK_TICKER, "QQQ", *config.SECTOR_ETFS]
    return sorted(set(names) | set(extra))


def _download(tickers: list[str]) -> pd.DataFrame:
    df = yf.download(tickers, period="730d", interval="60m", auto_adjust=True,
                     progress=False, threads=True, group_by="ticker", prepost=False)
    if df is None or df.empty:
        return pd.DataFrame()
    frames = []
    syms = tickers if len(tickers) > 1 else [tickers[0]]
    for sym in syms:
        try:
            sub = df[sym] if len(tickers) > 1 else df
        except KeyError:
            continue
        sub = sub.dropna(how="all")
        if sub.empty:
            continue
        sub = sub.reset_index()
        tcol = sub.columns[0]                       # 'Datetime' or 'index'
        sub = sub.rename(columns={tcol: "ts", "Open": "open", "High": "high",
                                  "Low": "low", "Close": "close", "Volume": "volume"})
        # normalize to tz-naive US/Eastern wall-clock
        ts = pd.to_datetime(sub["ts"], utc=True).dt.tz_convert("America/New_York")
        sub["ts"] = ts.dt.tz_localize(None)
        sub["ticker"] = sym
        frames.append(sub[["ticker", "ts", "open", "high", "low", "close", "volume"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ingest(verbose: bool = True) -> dict:
    con = connect()
    uni = universe(con)
    n_rows = 0
    for i in range(0, len(uni), BATCH):
        batch = uni[i:i + BATCH]
        try:
            frame = _download(batch)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  batch {i//BATCH} failed: {e}")
            continue
        if not frame.empty:
            con.execute("CREATE TEMP TABLE _b AS SELECT * FROM bars_hourly WHERE 0=1")
            con.executemany(
                "INSERT INTO _b VALUES (?,?,?,?,?,?,?)",
                frame.itertuples(index=False, name=None),
            )
            con.execute("DELETE FROM bars_hourly WHERE (ticker, ts) IN "
                        "(SELECT ticker, ts FROM _b)")
            con.execute("INSERT INTO bars_hourly SELECT * FROM _b")
            con.execute("DROP TABLE _b")
            n_rows += len(frame)
        if verbose:
            print(f"  batch {i//BATCH + 1}/{(len(uni)-1)//BATCH + 1}: {len(frame)} rows")
        time.sleep(1.0)
    total = con.execute("SELECT COUNT(*) FROM bars_hourly").fetchone()[0]
    cov = con.execute("SELECT COUNT(DISTINCT ticker) FROM bars_hourly").fetchone()[0]
    rng = con.execute("SELECT MIN(ts), MAX(ts) FROM bars_hourly").fetchone()
    con.close()
    if verbose:
        print(f"Hourly bars: {total} rows, {cov} tickers, {rng[0]} .. {rng[1]}")
    return {"rows": total, "tickers": cov}


if __name__ == "__main__":
    ingest()
