"""V4 data ingest: Alpaca minute bars + timestamped news (needs API keys).

Reads Alpaca keys from environment / a gitignored `.env` (see `.env.example`).
Ingests:
  * minute bars  -> `bars_minute`
  * news         -> the event bus (`events`, layer 'N', event_type 'news'),
                    public_time = the article's `created_at` (the moment it was
                    published — exactly what the leak-free replay needs).

This module is intentionally guarded: with no keys it prints setup instructions
and exits without touching the network. Paper/data endpoints only.

Docs: data.alpaca.markets/v2/stocks/bars (bars), /v1beta1/news (news).
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from .. import net  # noqa: F401 - verified TLS
from ..config import ROOT
from ..db import connect
from ..events.store import event_id, upsert_events

DATA_BARS = "https://data.alpaca.markets/v2/stocks/bars"
DATA_NEWS = "https://data.alpaca.markets/v1beta1/news"


def _load_env():
    """Populate os.environ from a gitignored .env (KEY=VALUE lines)."""
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _keys():
    _load_env()
    kid = os.environ.get("ALPACA_KEY")
    sec = os.environ.get("ALPACA_SECRET")
    return kid, sec


def _headers(kid, sec):
    return {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec,
            "User-Agent": net.DEFAULT_UA}


def _get(url, params, headers):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{q}", headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def ingest_bars(symbols: list[str], start: str, end: str, timeframe="1Min",
                verbose=True) -> dict:
    kid, sec = _keys()
    if not kid:
        _print_setup()
        return {"error": "no ALPACA keys"}
    h = _headers(kid, sec)
    con = connect()
    n = 0
    for i in range(0, len(symbols), 100):
        batch = ",".join(symbols[i:i + 100])
        page = None
        while True:
            params = {"symbols": batch, "timeframe": timeframe, "start": start,
                      "end": end, "limit": 10000, "adjustment": "split"}
            if page:
                params["page_token"] = page
            data = _get(DATA_BARS, params, h)
            bars = data.get("bars", {})
            rows = []
            for sym, arr in bars.items():
                for b in arr:
                    rows.append([sym, b["t"], b["o"], b["h"], b["l"], b["c"],
                                 b["v"], b.get("n"), b.get("vw")])
            if rows:
                con.execute("CREATE TEMP TABLE _bm AS SELECT * FROM bars_minute WHERE 0=1")
                con.executemany("INSERT INTO _bm VALUES (?,?,?,?,?,?,?,?,?)", rows)
                con.execute("DELETE FROM bars_minute WHERE (ticker, ts) IN (SELECT ticker, ts FROM _bm)")
                con.execute("INSERT INTO bars_minute SELECT * FROM _bm")
                con.execute("DROP TABLE _bm")
                n += len(rows)
            page = data.get("next_page_token")
            if not page:
                break
            time.sleep(0.3)
    con.close()
    if verbose:
        print(f"Alpaca bars: {n} minute rows for {len(symbols)} symbols")
    return {"rows": n}


def ingest_news(start: str, end: str, verbose=True) -> dict:
    kid, sec = _keys()
    if not kid:
        _print_setup()
        return {"error": "no ALPACA keys"}
    h = _headers(kid, sec)
    page, events, total = None, [], 0
    while True:
        params = {"start": start, "end": end, "limit": 50, "sort": "asc"}
        if page:
            params["page_token"] = page
        data = _get(DATA_NEWS, params, h)
        for a in data.get("news", []):
            total += 1
            ct = a.get("created_at")
            for sym in a.get("symbols", []):
                events.append({
                    "event_id": event_id("N", a["id"], sym),
                    "layer": "N", "event_type": "news", "ticker": sym,
                    "entity": a.get("source", ""), "direction": 0.0,
                    "magnitude": 0.5, "novelty": 1.0,
                    "effective_date": ct[:10] if ct else None,
                    "source_time": ct, "source_url": a.get("url"),
                    "payload": json.dumps({"headline": a.get("headline"),
                                           "source": a.get("source"),
                                           "symbols": a.get("symbols")}),
                })
        page = data.get("next_page_token")
        if not page:
            break
        time.sleep(0.3)
    upsert_events(events)
    if verbose:
        print(f"Alpaca news: {total} articles -> {len(events)} symbol-events (layer N)")
    return {"articles": total, "events": len(events)}


def _print_setup():
    print("Alpaca keys not found. Create a gitignored .env at repo root:\n"
          "  ALPACA_KEY=your_paper_key_id\n"
          "  ALPACA_SECRET=your_paper_secret\n"
          "  ALPACA_PAPER=true\n"
          "Get free paper keys at https://alpaca.markets (Paper Trading).")


if __name__ == "__main__":
    _print_setup()
