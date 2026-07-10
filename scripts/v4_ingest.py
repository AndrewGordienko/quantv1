"""Small real V4 ingest via Polygon free tier (2y minute + news).

Rate-limited (5 req/min) so we start with a compact liquid universe + a recent
window to validate the full pipeline, then expand later.
"""

from __future__ import annotations

from quantv1.ingest import polygon_data as P

# Starter tier: unlimited calls + 5y history. Stocks first (SPY = benchmark);
# ETFs already loaded. Ingester skips tickers already covered.
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "LLY", "UNH", "JPM"]
UNIVERSE = ["SPY", *STOCKS]
NEWS_TICKERS = STOCKS
START = "2024-07-15"
END = "2026-07-09"


def main():
    P.ingest_bars(UNIVERSE, START, END)
    for tk in NEWS_TICKERS:
        P.ingest_news(START, END, tickers=[tk])
    from quantv1.db import connect
    con = connect(read_only=True)
    nb = con.execute("SELECT COUNT(*) FROM bars_minute").fetchone()[0]
    nt = con.execute("SELECT COUNT(DISTINCT ticker) FROM bars_minute").fetchone()[0]
    nn = con.execute("SELECT COUNT(*) FROM events WHERE layer='N'").fetchone()[0]
    con.close()
    print(f"DONE: {nb} minute bars ({nt} tickers), {nn} news events")


if __name__ == "__main__":
    main()
