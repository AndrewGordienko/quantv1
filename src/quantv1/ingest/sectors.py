"""Ingest ticker -> sector / industry / market-cap into DuckDB `ticker_sectors`.

yfinance has no bulk sector endpoint, so this pulls `.info` per ticker. That is
slow and rate-limited, so we prioritize tickers by how often they were actually
traded (most-traded first) and tolerate failures — anything unresolved defaults
to sector "Unknown" downstream. Runs incrementally: already-known tickers are
skipped, so it fills in over successive daily runs.
"""

from __future__ import annotations

import time

import yfinance as yf

from ..db import connect

PAUSE = 0.3       # per-ticker courtesy pause


def pending_tickers(con, limit: int | None = None) -> list[str]:
    """Traded tickers without sector data yet, most-traded first."""
    rows = con.execute("""
        SELECT t.ticker, COUNT(*) AS n
        FROM trades t
        LEFT JOIN ticker_sectors s USING (ticker)
        WHERE s.ticker IS NULL
        GROUP BY t.ticker
        ORDER BY n DESC
    """).fetchall()
    tickers = [r[0] for r in rows]
    return tickers[:limit] if limit else tickers


def ingest(limit: int | None = None, verbose: bool = True) -> dict:
    con = connect()
    todo = pending_tickers(con, limit)
    got, failed = 0, 0
    for i, tk in enumerate(todo):
        sector = industry = None
        mcap = None
        try:
            info = yf.Ticker(tk).info
            sector = info.get("sector")
            industry = info.get("industry")
            mcap = info.get("marketCap")
        except Exception:  # noqa: BLE001 - transient/network/parse errors
            pass
        # Record even failures as "Unknown" so we don't re-fetch forever.
        con.execute(
            "INSERT OR REPLACE INTO ticker_sectors VALUES (?,?,?,?)",
            [tk, sector or "Unknown", industry or "Unknown", mcap],
        )
        if sector:
            got += 1
        else:
            failed += 1
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(todo)} resolved={got} unknown={failed}")
        time.sleep(PAUSE)
    total = con.execute("SELECT COUNT(*) FROM ticker_sectors").fetchone()[0]
    con.close()
    stats = {"attempted": len(todo), "resolved": got, "unknown": failed,
             "db_total": total}
    if verbose:
        print(f"Sectors: attempted={len(todo)} resolved={got} "
              f"unknown={failed} total={total}")
    return stats


if __name__ == "__main__":
    import sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    ingest(limit=lim)
