"""Ingest congressional committee assignments -> DuckDB `members`.

Source: the unitedstates/congress-legislators project (public-domain YAML/JSON).
We join current legislators to their committee memberships, map each committee's
legislative jurisdiction to the GICS sectors it plausibly oversees, and store
per-member the committee list + the union of jurisdiction sectors.

The payoff downstream is one feature: *is this member trading a stock in a
sector their committee has authority over?* (e.g. an Armed Services member
buying a defense contractor). That is the single most interpretable "why did
they trade this" signal in the whole system.

Limitation (documented): the feed covers CURRENT members only. Historical
members who traded and later left Congress get no committee data and fall back
to "no committee match". The live portfolio uses current members, so this is
acceptable; it only weakens the historical backtest's committee feature.
"""

from __future__ import annotations

import json
import re
import urllib.request

from ..db import connect

_UA = {"User-Agent": "quantv1-ingest"}
BASE = "https://unitedstates.github.io/congress-legislators"

# Map a committee (by name keyword) to the sectors its jurisdiction touches.
# Sector names follow YAHOO/yfinance's taxonomy (what `ticker_sectors` stores),
# NOT GICS — e.g. "Healthcare", "Financial Services", "Consumer Cyclical".
# Hand-authored; deliberately broad — false negatives hurt more than mild
# false positives.
COMMITTEE_SECTORS: dict[str, list[str]] = {
    "armed services":       ["Industrials"],                       # defense primes
    "energy":               ["Energy", "Utilities"],
    "natural resources":    ["Energy", "Basic Materials", "Utilities"],
    "financial services":   ["Financial Services", "Real Estate"],
    "banking":              ["Financial Services", "Real Estate"],
    "health":               ["Healthcare"],
    "energy and commerce":  ["Healthcare", "Communication Services", "Technology",
                             "Energy", "Consumer Cyclical"],
    "commerce":             ["Communication Services", "Technology", "Industrials",
                             "Consumer Cyclical"],
    "science":              ["Technology", "Industrials", "Healthcare"],
    "agriculture":          ["Consumer Defensive", "Basic Materials"],
    "transportation":       ["Industrials"],
    "homeland security":    ["Industrials", "Technology"],
    "veterans":             ["Healthcare"],
    "small business":       [],
    "judiciary":            ["Technology", "Communication Services"],
    "intelligence":         ["Technology", "Industrials"],
    "foreign affairs":      ["Industrials", "Energy"],
    "ways and means":       ["Financial Services", "Healthcare"],  # tax/trade
    "appropriations":       [],                             # touches everything
    "finance":              ["Financial Services", "Healthcare"],  # senate finance
}


def _grab(name: str):
    req = urllib.request.Request(f"{BASE}/{name}", headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _member_key(name: str) -> str:
    """Match the normalization used in stockwatcher.member_key."""
    n = name.lower()
    n = re.sub(r"[.,]", " ", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv|mr|mrs|ms|dr|hon)\b", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _sectors_for(committee_name: str) -> list[str]:
    low = committee_name.lower()
    sectors: set[str] = set()
    for kw, secs in COMMITTEE_SECTORS.items():
        if kw in low:
            sectors.update(secs)
    return sorted(sectors)


def ingest(verbose: bool = True) -> dict:
    legislators = _grab("legislators-current.json")
    membership = _grab("committee-membership-current.json")   # thomas_id -> [members]
    committees = {c["thomas_id"]: c for c in _grab("committees-current.json")}

    # bioguide -> list of committee display names
    bioguide_committees: dict[str, list[str]] = {}
    for thomas_id, members in membership.items():
        comm = committees.get(thomas_id)
        if not comm:
            continue
        for m in members:
            bioguide_committees.setdefault(m["bioguide"], []).append(comm["name"])

    rows = []
    for leg in legislators:
        bio = leg["id"].get("bioguide")
        full = leg["name"].get("official_full") or (
            f"{leg['name'].get('first','')} {leg['name'].get('last','')}")
        term = leg["terms"][-1]
        comm_names = bioguide_committees.get(bio, [])
        sectors: set[str] = set()
        for cn in comm_names:
            sectors.update(_sectors_for(cn))
        rows.append({
            "member_key": _member_key(full),
            "member": full,
            "chamber": "senate" if term.get("type") == "sen" else "house",
            "party": term.get("party", ""),
            "state": term.get("state", ""),
            "committees": json.dumps({"committees": comm_names,
                                      "jurisdiction_sectors": sorted(sectors)}),
        })

    con = connect()
    con.execute("CREATE TEMP TABLE _m AS SELECT * FROM members WHERE 0=1")
    con.executemany(
        "INSERT INTO _m VALUES (?,?,?,?,?,?)",
        [[r["member_key"], r["member"], r["chamber"], r["party"], r["state"],
          r["committees"]] for r in rows],
    )
    con.execute("DELETE FROM members WHERE member_key IN (SELECT member_key FROM _m)")
    con.execute("INSERT INTO members SELECT * FROM _m")
    con.execute("DROP TABLE _m")

    # How many trading members did we actually match to committee data?
    matched = con.execute("""
        SELECT COUNT(DISTINCT t.member_key)
        FROM trades t JOIN members m USING (member_key)
    """).fetchone()[0]
    traders = con.execute("SELECT COUNT(DISTINCT member_key) FROM trades").fetchone()[0]
    con.close()

    stats = {"legislators": len(rows), "trading_members_matched": matched,
             "trading_members_total": traders}
    if verbose:
        print(f"Members: {len(rows)} current legislators; "
              f"matched {matched}/{traders} trading members to committee data")
    return stats


if __name__ == "__main__":
    ingest()
