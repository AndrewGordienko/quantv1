"""Actor registry + first (proxy) actor-event extraction from existing news.

The clean actor-event sources are Fed/central-bank comms and earnings calls
(exact timestamps, clear exposure) — those come next. As a tractable FIRST step
that needs no new scraper, we extract actor MENTIONS from the Polygon news
headlines already in the event bus: for a curated set of high-information actors
tied to our universe, a headline naming the actor becomes an actor_event
inheriting that news event's public_time, tickers and catalyst_id.

This is deliberately a proxy (a mention is not the actor speaking), good enough to
run the first descriptive B1-vs-B2 test (does conditioning on WHO adds anything
over generic news) before investing in transcript-grade sources.
"""

from __future__ import annotations

import hashlib
import json
import re

from ..db import connect

# Curated ~20 high-information actors tied to the loaded universe + macro.
# (authority is a coarse prior, refined later by measured impact.)
ACTORS = [
    ("powell", "Jerome Powell", "central_banker", "SPY", 0.95, ["powell", "fed chair", "federal reserve chair"]),
    ("musk", "Elon Musk", "ceo", "TSLA", 0.9, ["musk"]),
    ("huang", "Jensen Huang", "ceo", "NVDA", 0.8, ["jensen huang", "huang"]),
    ("cook", "Tim Cook", "ceo", "AAPL", 0.7, ["tim cook"]),
    ("nadella", "Satya Nadella", "ceo", "MSFT", 0.7, ["nadella"]),
    ("jassy", "Andy Jassy", "ceo", "AMZN", 0.6, ["jassy"]),
    ("pichai", "Sundar Pichai", "ceo", "GOOGL", 0.7, ["pichai"]),
    ("zuckerberg", "Mark Zuckerberg", "ceo", "META", 0.75, ["zuckerberg"]),
    ("dimon", "Jamie Dimon", "ceo", "JPM", 0.7, ["jamie dimon", "dimon"]),
    ("su", "Lisa Su", "ceo", "AMD", 0.65, ["lisa su"]),
    ("hassett", "Kevin Hassett", "official", "SPY", 0.6, ["hassett"]),
    ("bessent", "Scott Bessent", "official", "SPY", 0.7, ["bessent"]),
    ("gensler", "SEC Chair", "regulator", "SPY", 0.6, ["sec chair", "gensler", "atkins"]),
]


def _aid(actor, ticker, catalyst_id, public_time):
    return hashlib.sha1(f"{actor}|{ticker}|{catalyst_id}|{public_time}".encode()).hexdigest()[:20]


def register(verbose=True):
    con = connect()
    con.execute("DELETE FROM actors")
    con.executemany("INSERT INTO actors VALUES (?,?,?,?,?,?)",
                    [[a, n, r, o, au, json.dumps(al)] for a, n, r, o, au, al in ACTORS])
    # time-valid 'leads' edges (open-ended; refine dates later)
    con.execute("DELETE FROM actor_relationships")
    con.executemany("INSERT INTO actor_relationships VALUES (?,?,?,?,?,?,?)",
                    [[a, "leads" if r != "regulator" else "regulates", o, "2020-01-01", None,
                      "curated", None] for a, n, r, o, au, al in ACTORS])
    con.close()
    if verbose:
        print(f"Registered {len(ACTORS)} actors + relationships")


def extract_from_news(verbose=True) -> dict:
    """Scan N-layer news headlines for curated actor mentions -> actor_events."""
    con = connect()
    rows = con.execute("""
        SELECT ticker, source_time, catalyst_id, payload FROM events
        WHERE layer='N' AND ticker IS NOT NULL AND source_time IS NOT NULL
    """).fetchall()
    patterns = [(a, re.compile(r"\b(" + "|".join(re.escape(x) for x in al) + r")\b", re.I))
                for a, n, r, o, au, al in ACTORS]
    seen, out = set(), []
    for tk, st, cat, payload in rows:
        try:
            title = json.loads(payload).get("title", "") or ""
        except (TypeError, ValueError):
            continue
        for actor, pat in patterns:
            if pat.search(title):
                aeid = _aid(actor, tk, cat, st)
                if (aeid, tk) in seen:
                    continue
                seen.add((aeid, tk))
                out.append([aeid, actor, tk, str(st), "news_mention", title[:200],
                            cat, "polygon_news", None])
    con.execute("DELETE FROM actor_events")
    con.executemany("INSERT INTO actor_events VALUES (?,?,?,?,?,?,?,?,?)", out)
    by_actor = dict(con.execute("""
        SELECT actor_id, COUNT(*) FROM actor_events GROUP BY 1 ORDER BY 2 DESC
    """).fetchall())
    con.close()
    if verbose:
        print(f"Actor-events (news mentions): {len(out)} rows")
        for a, c in list(by_actor.items())[:12]:
            print(f"  {a:12s} {c}")
    return {"actor_events": len(out), "by_actor": by_actor}


if __name__ == "__main__":
    register()
    extract_from_news()
