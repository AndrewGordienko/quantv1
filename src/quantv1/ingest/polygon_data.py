"""V4 data ingest: Polygon.io minute bars + timestamped news (Canada-friendly).

Polygon is a market-DATA vendor (not a broker), so it works from Canada. Reads
POLYGON_API_KEY from a gitignored `.env`. Writes to the SAME tables the Alpaca
ingester would (`bars_minute` and the `events` bus, layer 'N'), so the leak-free
replay engine is vendor-agnostic.

  * minute bars  -> `bars_minute`   (aggregates v2 endpoint, per ticker)
  * news         -> `events` layer 'N', public_time = article `published_utc`

Guarded: with no key it prints setup instructions and touches no network.
Free tier: ~5 calls/min, 2y history — enough to prototype; paid for full history.
Docs: polygon.io/docs (aggregates, reference/news).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime, timezone

from .. import net  # noqa: F401 - verified TLS
from ..config import ROOT
from ..db import connect
from ..events.store import event_id, upsert_events

_BARS = "https://api.polygon.io/v2/aggs/ticker/{tk}/range/1/minute/{a}/{b}"
_NEWS = "https://api.polygon.io/v2/reference/news"
_RATE_SLEEP = 0.05    # Starter tier: unlimited calls; tiny courtesy pause
_BACKOFF = 3.0        # on transient error


def _key():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    return os.environ.get("POLYGON_API_KEY")


def _get(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": net.DEFAULT_UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001 - rate limit / transient
            time.sleep(_BACKOFF)
    return None


def ingest_bars(symbols: list[str], start: str, end: str, verbose=True) -> dict:
    key = _key()
    if not key:
        _setup(); return {"error": "no POLYGON_API_KEY"}
    con = connect()
    # skip tickers already covered (avoid re-fetching / re-checking millions of rows)
    have = dict(con.execute("SELECT ticker, COUNT(*) FROM bars_minute GROUP BY 1").fetchall())
    n = 0
    for tk in symbols:
        if have.get(tk, 0) > 100_000:                # already ingested this ticker
            if verbose:
                print(f"  {tk}: already have {have[tk]} bars, skipping")
            continue
        url = (_BARS.format(tk=tk, a=start, b=end) +
               f"?adjusted=true&sort=asc&limit=50000&apiKey={key}")
        while url:
            data = _get(url)
            if not data:
                break
            rows = []
            for b in data.get("results", []):
                ts = datetime.fromtimestamp(b["t"] / 1000, tz=timezone.utc).replace(tzinfo=None)
                rows.append([tk, ts, b.get("o"), b.get("h"), b.get("l"), b.get("c"),
                             b.get("v"), b.get("n"), b.get("vw")])
            if rows:
                # bulk load: dupes are skipped by PK; no O(n) delete scan per page
                con.executemany(
                    "INSERT INTO bars_minute VALUES (?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT DO NOTHING", rows)
                n += len(rows)
            nxt = data.get("next_url")
            url = f"{nxt}&apiKey={key}" if nxt else None
            time.sleep(_RATE_SLEEP)
        if verbose:
            print(f"  {tk}: {n} rows so far")
    con.close()
    if verbose:
        print(f"Polygon bars: {n} minute rows for {len(symbols)} symbols")
    return {"rows": n}


def ingest_news(start: str, end: str, tickers: list[str] | None = None,
                verbose=True) -> dict:
    key = _key()
    if not key:
        _setup(); return {"error": "no POLYGON_API_KEY"}
    base = (f"{_NEWS}?published_utc.gte={start}&published_utc.lte={end}"
            f"&order=asc&limit=1000&apiKey={key}")
    if tickers:
        base += "&ticker=" + ",".join(tickers[:1])   # polygon news filters one ticker
    url, total, events = base, 0, []
    while url:
        data = _get(url)
        if not data:
            break
        for a in data.get("results", []):
            total += 1
            pub = a.get("published_utc")
            for sym in a.get("tickers", []):
                events.append({
                    "event_id": event_id("N", a["id"], sym),
                    "layer": "N", "event_type": "news", "ticker": sym,
                    "entity": (a.get("publisher") or {}).get("name", ""),
                    "direction": 0.0, "magnitude": 0.5, "novelty": 1.0,
                    "effective_date": pub[:10] if pub else None,
                    "source_time": pub, "source_url": a.get("article_url"),
                    "payload": json.dumps({"title": a.get("title"),
                                           "publisher": (a.get("publisher") or {}).get("name"),
                                           "tickers": a.get("tickers"),
                                           "insights": a.get("insights")}),
                })
        nxt = data.get("next_url")
        url = f"{nxt}&apiKey={key}" if nxt else None
        time.sleep(_RATE_SLEEP)
    upsert_events(events)
    if verbose:
        print(f"Polygon news: {total} articles -> {len(events)} symbol-events (layer N)")
    return {"articles": total, "events": len(events)}


def _setup():
    print("Polygon key not found. Create a gitignored .env at repo root:\n"
          "  POLYGON_API_KEY=your_key\n"
          "Free key (2y history, 5 req/min): https://polygon.io/dashboard/api-keys")


if __name__ == "__main__":
    _setup()
