"""Entity resolution: SEC ticker <-> CIK <-> legal name.

The methodology review is right that entity resolution is foundational — probably
more valuable than the first ML model, because every downstream layer (insider
Form 4, 8-K events, government contracts, lobbying) keys off the company's CIK,
not its ticker. This ingests SEC's public, free `company_tickers.json` — a
deterministic, authoritative mapping — into `sec_entities`.

Later layers resolve a congress-traded ticker to its CIK here, then pull that
company's EDGAR filings; contractor/subsidiary → parent resolution is a separate,
harder step handled with verified mappings on top of this anchor.
"""

from __future__ import annotations

import json
import urllib.request

from ..db import connect

URL = "https://www.sec.gov/files/company_tickers.json"
# SEC requires a descriptive UA with contact info for its APIs.
_UA = {"User-Agent": "quantv1 research (andrew.gordienko05@gmail.com)"}


def fetch() -> list[dict]:
    req = urllib.request.Request(URL, headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = json.load(r)
    # payload is {"0": {cik_str, ticker, title}, "1": {...}, ...}
    rows = []
    for v in raw.values():
        tk = str(v.get("ticker", "")).upper().strip()
        if not tk:
            continue
        rows.append({"ticker": tk, "cik": str(v["cik_str"]).zfill(10),
                     "title": v.get("title", "")})
    return rows


def ingest(verbose: bool = True) -> dict:
    rows = fetch()
    con = connect()
    con.execute("DELETE FROM sec_entities")
    # keep one row per ticker (first CIK wins; dupes are rare share classes)
    seen = set()
    payload = []
    for r in rows:
        if r["ticker"] in seen:
            continue
        seen.add(r["ticker"])
        payload.append([r["ticker"], r["cik"], r["title"]])
    con.executemany("INSERT INTO sec_entities VALUES (?,?,?)", payload)

    # coverage against our traded universe
    covered = con.execute("""
        SELECT COUNT(DISTINCT t.ticker)
        FROM trades t JOIN sec_entities s ON t.ticker = s.ticker
    """).fetchone()[0]
    total = con.execute("SELECT COUNT(DISTINCT ticker) FROM trades").fetchone()[0]
    n = con.execute("SELECT COUNT(*) FROM sec_entities").fetchone()[0]
    con.close()
    if verbose:
        print(f"SEC entities: {n} tickers mapped to CIK; "
              f"resolved {covered}/{total} traded tickers")
    return {"sec_rows": n, "resolved": covered, "traded_total": total}


if __name__ == "__main__":
    ingest()
