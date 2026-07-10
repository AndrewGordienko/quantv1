"""G-layer ingest: Federal Register final RULES as events.

Reachable, free, no key. Regulations act on industries/agencies more than named
companies, so these are stored as sector-tagged slow-context events: the agency
maps to affected sectors, and `publication_date` IS the true public availability
date (unlike USAspending, no lag correction needed). Company-level resolution of
a rule's text is a later LLM-extraction step; for now the event carries the
agency + inferred sectors so a sector-level study is possible.

We prioritize "significant" rules (economically/politically material).
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime

import numpy as np

from .. import net  # noqa: F401 - verified TLS via OS trust store
from ..db import connect
from ..events.store import event_id, upsert_events

_UA = {"User-Agent": net.DEFAULT_UA}
_BASE = "https://www.federalregister.gov/api/v1/documents.json"

# Agency-name keyword -> affected Yahoo sectors (coarse but honest slow-context).
AGENCY_SECTORS: dict[str, list[str]] = {
    "defense": ["Industrials"],
    "energy department": ["Energy", "Utilities"],
    "environmental protection": ["Energy", "Utilities", "Basic Materials"],
    "health": ["Healthcare"],
    "food and drug": ["Healthcare"],
    "transportation": ["Industrials"],
    "federal communications": ["Communication Services", "Technology"],
    "agriculture": ["Consumer Defensive", "Basic Materials"],
    "treasury": ["Financial Services"],
    "securities and exchange": ["Financial Services"],
    "federal reserve": ["Financial Services"],
    "interior": ["Energy", "Basic Materials"],
    "homeland security": ["Industrials", "Technology"],
    "labor": [],
    "commerce": ["Technology", "Industrials", "Consumer Cyclical"],
}


def _sectors_for(agencies: list[str]) -> list[str]:
    secs: set[str] = set()
    for a in agencies:
        low = (a or "").lower()
        for kw, s in AGENCY_SECTORS.items():
            if kw in low:
                secs.update(s)
    return sorted(secs)


def _fetch_page(year: int, page: int) -> dict | None:
    params = (
        f"?per_page=100&page={page}&order=oldest"
        f"&conditions[type][]=RULE"
        f"&conditions[publication_date][gte]={year}-01-01"
        f"&conditions[publication_date][lte]={year}-12-31"
        "&fields[]=title&fields[]=publication_date&fields[]=agencies"
        "&fields[]=significant&fields[]=abstract&fields[]=document_number"
        "&fields[]=html_url"
    )
    req = urllib.request.Request(_BASE + params, headers=_UA)
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=45))
        except Exception:  # noqa: BLE001
            time.sleep(1.0 * (attempt + 1))
    return None


def ingest(start_year: int = 2016, end_year: int = 2026,
           significant_only: bool = True, verbose: bool = True) -> dict:
    total, written = 0, 0
    for year in range(start_year, end_year + 1):
        page = 1
        while page <= 20:                            # FR API caps deep paging
            data = _fetch_page(year, page)
            if not data or not data.get("results"):
                break
            rows = []
            for d in data["results"]:
                total += 1
                if significant_only and not d.get("significant"):
                    continue
                agencies = [a.get("name") for a in d.get("agencies", []) if a.get("name")]
                sectors = _sectors_for(agencies)
                pd_ = d.get("publication_date")
                try:
                    y, m, dd = map(int, pd_.split("-"))
                    pub = datetime(y, m, dd)
                except (ValueError, AttributeError):
                    continue
                rows.append({
                    "event_id": event_id("G", "FR", d.get("document_number")),
                    "layer": "G", "event_type": "reg_rule",
                    "ticker": None,                  # sector-level, not per-company
                    "entity": agencies[0] if agencies else "",
                    "direction": 0.0,               # sign unknown without text analysis
                    "magnitude": 1.0 if d.get("significant") else 0.5,
                    "novelty": 1.0,
                    "effective_date": pub.date(),
                    "source_time": pub,             # publication_date IS public
                    "source_url": d.get("html_url"),
                    "payload": json.dumps({"title": d.get("title"),
                                           "agencies": agencies, "sectors": sectors,
                                           "significant": bool(d.get("significant"))}),
                })
            if rows:
                upsert_events(rows)
                written += len(rows)
            if len(data["results"]) < 100:
                break
            page += 1
            time.sleep(0.3)
        if verbose:
            print(f"  {year}: {written} significant rules so far")

    con = connect(read_only=True)
    n = con.execute("SELECT COUNT(*) FROM events WHERE event_type='reg_rule'").fetchone()[0]
    con.close()
    if verbose:
        print(f"Federal Register: {n} rule events (scanned {total})")
    return {"reg_events": n, "scanned": total}


if __name__ == "__main__":
    ingest()
