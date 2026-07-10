"""Deduplicate news events into CATALYSTS.

One market-moving story appears as many rows: the same article is tagged with
several tickers (one N-event each), and updates/revisions of the story arrive as
separate articles. Both inflate event counts and, worse, make same-story trades
look independent. This collapses them:

  catalyst_id = hash( normalized-headline , calendar-day )

so intraday updates of the same headline merge into one catalyst, and the union
of their tickers is the catalyst's CROSS-TICKER membership. `earliest_public_time`
is the first time we could have acted. Every N event is stamped with catalyst_id.

Different calendar days are treated as different catalysts (a story continuing the
next session is a new actionable event). This is deliberately simple and
point-in-time; embedding-based clustering can refine it later.
"""

from __future__ import annotations

import hashlib
import json
import re

import duckdb

from ..config import DB_PATH
from ..db import connect

_WORD = re.compile(r"[^a-z0-9 ]+")


def _norm(title: str) -> str:
    t = _WORD.sub(" ", (title or "").lower())
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(t.split()[:10])            # first ~10 tokens (tolerate minor edits)


def _catalyst_id(headline_key: str, day: str) -> str:
    return hashlib.sha1(f"{headline_key}|{day}".encode()).hexdigest()[:16]


def build(verbose: bool = True) -> dict:
    # add the column if missing (migration)
    raw = duckdb.connect(str(DB_PATH))
    cols = [r[1] for r in raw.execute("PRAGMA table_info('events')").fetchall()]
    if "catalyst_id" not in cols:
        raw.execute("ALTER TABLE events ADD COLUMN catalyst_id VARCHAR")
    raw.close()

    con = connect()
    rows = con.execute("""
        SELECT event_id, ticker, source_time, payload FROM events WHERE layer='N'
    """).fetchall()
    if not rows:
        con.close()
        return {"catalysts": 0, "events": 0}

    cat_of_event = {}
    catalysts: dict[str, dict] = {}
    for eid, tk, st, payload in rows:
        try:
            title = json.loads(payload).get("title", "")
        except (TypeError, ValueError):
            title = ""
        st = str(st)
        day = st[:10]
        cid = _catalyst_id(_norm(title), day)
        cat_of_event[eid] = cid
        c = catalysts.setdefault(cid, {"earliest": st, "headline": title,
                                       "tickers": set(), "n_events": 0})
        c["tickers"].add(tk)
        c["n_events"] += 1
        if st < c["earliest"]:
            c["earliest"] = st
            c["headline"] = title

    # stamp events with catalyst_id
    con.execute("CREATE TEMP TABLE _cm(event_id VARCHAR, catalyst_id VARCHAR)")
    con.executemany("INSERT INTO _cm VALUES (?,?)", list(cat_of_event.items()))
    con.execute("UPDATE events SET catalyst_id = _cm.catalyst_id "
                "FROM _cm WHERE events.event_id = _cm.event_id")
    con.execute("DROP TABLE _cm")

    # (re)build the catalysts table
    con.execute("DELETE FROM news_catalysts")
    payload = []
    for cid, c in catalysts.items():
        tickers = sorted(c["tickers"])
        primary = max(tickers, key=lambda t: 0)  # deterministic; refine w/ salience later
        payload.append([cid, c["earliest"], c["headline"][:200], tickers[0],
                        json.dumps(tickers), len(tickers), c["n_events"]])
    con.executemany("INSERT INTO news_catalysts VALUES (?,?,?,?,?,?,?)", payload)
    n_cat = con.execute("SELECT COUNT(*) FROM news_catalysts").fetchone()[0]
    n_multi = con.execute("SELECT COUNT(*) FROM news_catalysts WHERE n_tickers > 1").fetchone()[0]
    con.close()

    stats = {"catalysts": n_cat, "events": len(rows), "multi_ticker": n_multi,
             "dedup_ratio": round(len(rows) / max(n_cat, 1), 2)}
    if verbose:
        print(f"Catalysts: {len(rows)} N-events -> {n_cat} catalysts "
              f"({stats['dedup_ratio']}x dedup), {n_multi} span multiple tickers")
    return stats


if __name__ == "__main__":
    build()
