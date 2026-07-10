"""V4 ingest via the reliable pipeline: FETCH (Polygon->Parquet) then LOAD (bulk).

Resumable + watermarked + completeness-audited. News is separate (smaller).
"""

from __future__ import annotations

from quantv1.v4 import data_pipeline as DP
from quantv1.ingest import polygon_data as P

STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "LLY", "UNH", "JPM",
          "AVGO", "AMD", "NFLX", "CRM", "BAC", "XOM", "COST", "WMT"]
ETFS = ["SPY", "QQQ", "XLK", "XLV", "XLE", "XLF", "XLI", "XLY"]
UNIVERSE = ETFS + STOCKS
START = "2024-07-15"
END = "2026-07-09"


def main():
    DP.fetch(UNIVERSE, START, END)      # network -> parquet (no DB lock)
    DP.load()                           # single bulk parquet -> bars_minute
    DP.completeness()
    for tk in STOCKS:                   # news for the tradeable names
        P.ingest_news(START, END, tickers=[tk], verbose=False)
    from quantv1.db import connect
    c = connect(read_only=True)
    print("news events:", c.execute("SELECT COUNT(*) FROM events WHERE layer='N'").fetchone()[0])
    c.close()


if __name__ == "__main__":
    main()
