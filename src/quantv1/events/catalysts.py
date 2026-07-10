"""Point-in-time, versioned news-catalyst construction.

The clustering unit is an article revision, not a ticker-event row.  Exact
provider article/URL identity is resolved first.  Distinct articles are merged
only when their full headlines are near duplicates inside a bounded time window
and they share a resolved entity (including a tagged ticker).

Membership is append-only.  ``events.catalyst_id`` is a convenience pointer to
the current algorithm's assignment; historical assignments remain in
``catalyst_events``.  A ticker becomes usable at its own
``catalyst_assets.first_link_public_time``, never at the catalyst's earliest
timestamp.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
import hashlib
import json
import re
from typing import Iterable

import pandas as pd

from ..db import connect

CLUSTER_VERSION = "v3-exact-url-neardup-entity-12h"
CLUSTER_WINDOW = timedelta(hours=12)
TOKEN_JACCARD_MIN = 0.72
SEQUENCE_SIM_MIN = 0.88
TOKEN_CONTAINMENT_MIN = 0.75

_NON_WORD = re.compile(r"[^a-z0-9]+")
_CAPITALIZED = re.compile(
    r"\b(?:[A-Z][a-z]{2,}|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]{2,}|[A-Z]{2,})){0,3}\b"
)
_GENERIC_ENTITIES = {
    "after", "amid", "before", "breaking", "company", "exclusive", "federal",
    "global", "markets", "new", "news", "president", "report", "says", "stocks",
    "the", "update", "wall street",
}


def _full_norm(text: str) -> str:
    """Normalize the whole headline; never truncate to a generic prefix."""
    return " ".join(_NON_WORD.sub(" ", (text or "").lower()).split())


def _payload(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _headline(payload: dict) -> str:
    return str(payload.get("title") or payload.get("headline") or "").strip()


def _source_identity(event_id: str, source_url: str | None, publisher: str,
                     public_time: datetime, headline: str, payload: dict) -> str:
    """Stable exact identity before any fuzzy clustering.

    New ingesters persist the provider's article ID.  Existing data falls back
    to the exact URL.  The final fallback intentionally includes timestamp and
    full headline so unrelated anonymous articles cannot collapse together.
    """
    provider_id = payload.get("article_id") or payload.get("source_article_id")
    provider = str(payload.get("provider") or payload.get("publisher") or
                   payload.get("source") or publisher or "unknown").strip().lower()
    if provider_id not in (None, ""):
        return f"provider:{provider}:{provider_id}"
    if source_url:
        return f"url:{source_url.strip()}"
    fallback = f"{provider}|{public_time.isoformat()}|{_full_norm(headline)}"
    if not _full_norm(headline):
        fallback += f"|{event_id}"
    return f"fallback:{fallback}"


def _entities(headline: str, payload: dict, ticker: str | None) -> frozenset[str]:
    values: set[str] = set()
    for key in ("tickers", "symbols"):
        for value in payload.get(key) or []:
            if value:
                values.add(f"ticker:{str(value).upper()}")
    if ticker:
        values.add(f"ticker:{ticker.upper()}")
    for match in _CAPITALIZED.finditer(headline or ""):
        value = _full_norm(match.group())
        if value and value not in _GENERIC_ENTITIES:
            values.add(f"name:{value}")
    return frozenset(values)


@dataclass
class ArticleRevision:
    identity: str
    revision_id: str
    public_time: datetime
    headline: str
    tokens: frozenset[str]
    entities: set[str] = field(default_factory=set)
    event_ids: list[str] = field(default_factory=list)
    ticker_links: list[tuple[str, str, datetime]] = field(default_factory=list)


class _UnionFind:
    def __init__(self, revisions: list[ArticleRevision]):
        n = len(revisions)
        self.parent = list(range(n))
        self.minimum = [revision.public_time for revision in revisions]
        self.maximum = [revision.public_time for revision in revisions]

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        minimum = min(self.minimum[ra], self.minimum[rb])
        maximum = max(self.maximum[ra], self.maximum[rb])
        if maximum - minimum > CLUSTER_WINDOW:
            return False
        self.parent[rb] = ra
        self.minimum[ra] = minimum
        self.maximum[ra] = maximum
        return True


def _near_duplicate(a: ArticleRevision, b: ArticleRevision) -> bool:
    if not (a.entities & b.entities):
        return False
    if not a.tokens or not b.tokens:
        return False
    intersection = len(a.tokens & b.tokens)
    union = len(a.tokens | b.tokens)
    jaccard = intersection / union
    containment = intersection / min(len(a.tokens), len(b.tokens))
    sequence = SequenceMatcher(None, _full_norm(a.headline),
                               _full_norm(b.headline)).ratio()
    return jaccard >= TOKEN_JACCARD_MIN or (
        sequence >= SEQUENCE_SIM_MIN and containment >= TOKEN_CONTAINMENT_MIN
    )


def _article_revisions(rows: Iterable[tuple]) -> list[ArticleRevision]:
    by_revision: dict[str, ArticleRevision] = {}
    for event_id, ticker, public_time, source_url, publisher, raw_payload in rows:
        if public_time is None:
            continue
        payload = _payload(raw_payload)
        headline = _headline(payload)
        identity = _source_identity(event_id, source_url, publisher, public_time,
                                    headline, payload)
        content_hash = hashlib.sha1(_full_norm(headline).encode()).hexdigest()[:16]
        revision_id = hashlib.sha1(
            f"{identity}|{public_time.isoformat()}|{content_hash}".encode()
        ).hexdigest()[:20]
        revision = by_revision.get(revision_id)
        if revision is None:
            revision = ArticleRevision(
                identity=identity,
                revision_id=revision_id,
                public_time=public_time,
                headline=headline,
                tokens=frozenset(_full_norm(headline).split()),
            )
            by_revision[revision_id] = revision
        revision.entities.update(_entities(headline, payload, ticker))
        revision.event_ids.append(event_id)
        if ticker:
            revision.ticker_links.append((ticker, event_id, public_time))
    return sorted(by_revision.values(), key=lambda r: (r.public_time, r.revision_id))


def cluster_rows(rows: Iterable[tuple]) -> tuple[list[ArticleRevision], list[list[int]]]:
    """Pure clustering core, exposed for deterministic leakage regression tests."""
    revisions = _article_revisions(rows)
    uf = _UnionFind(revisions)

    # Exact provider article/URL identity always wins over fuzzy matching.
    first_by_identity: dict[str, int] = {}
    for i, revision in enumerate(revisions):
        if revision.identity in first_by_identity:
            if not uf.union(i, first_by_identity[revision.identity]):
                first_by_identity[revision.identity] = i
        else:
            first_by_identity[revision.identity] = i

    # Candidate generation is entity-blocked, then constrained by elapsed time.
    # This avoids O(n^2) comparisons and makes entity overlap structural.
    recent_by_entity: dict[str, list[int]] = defaultdict(list)
    for i, revision in enumerate(revisions):
        cutoff = revision.public_time - CLUSTER_WINDOW
        candidates: set[int] = set()
        for entity in revision.entities:
            recent = recent_by_entity[entity]
            while recent and revisions[recent[0]].public_time < cutoff:
                recent.pop(0)
            candidates.update(recent)
        for j in candidates:
            if revision.identity != revisions[j].identity and _near_duplicate(revision, revisions[j]):
                uf.union(i, j)
        for entity in revision.entities:
            recent_by_entity[entity].append(i)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(revisions)):
        groups[uf.find(i)].append(i)
    return revisions, sorted(groups.values(), key=lambda g: revisions[g[0]].public_time)


def _data_snapshot_hash(rows: list[tuple]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update("|".join("" if value is None else str(value) for value in row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _catalyst_id(revisions: list[ArticleRevision], group: list[int], build_id: str) -> str:
    anchor = min((revisions[i] for i in group),
                 key=lambda r: (r.public_time, r.identity, r.revision_id))
    raw = f"{build_id}|{CLUSTER_VERSION}|{anchor.identity}|{anchor.public_time.isoformat()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def latest_completed_build(con, cluster_version: str = CLUSTER_VERSION) -> str:
    row = con.execute("""
        SELECT build_id FROM catalyst_builds
        WHERE cluster_version=? AND status='COMPLETE'
        ORDER BY created_at DESC,build_id DESC LIMIT 1
    """, [cluster_version]).fetchone()
    if not row:
        raise ValueError(f"no completed catalyst build for {cluster_version}")
    return row[0]


def assets_as_of(con, as_of, *, build_id: str):
    """Return only catalyst assets whose own public-link gate has opened."""
    return con.execute("""
        SELECT ca.catalyst_id, ca.ticker, ca.first_link_public_time,
               ca.source_event_id
        FROM catalyst_assets ca
        WHERE ca.build_id=? AND ca.first_link_public_time <= ?
        ORDER BY ca.first_link_public_time, ca.catalyst_id, ca.ticker
    """, [build_id, as_of]).df()


def build(verbose: bool = True) -> dict:
    con = connect()
    rows = con.execute("""
        SELECT event_id, ticker, source_time, source_url, entity, payload
        FROM events
        WHERE layer='N' AND source_time IS NOT NULL
        ORDER BY source_time, event_id
    """).fetchall()
    if not rows:
        con.close()
        return {
            "raw_ticker_event_rows": 0, "raw_articles": 0,
            "unique_article_revisions": 0, "catalysts": 0,
            "catalyst_ticker_observations": 0,
            "cluster_version": CLUSTER_VERSION,
        }

    snapshot_hash = _data_snapshot_hash(rows)
    build_id = hashlib.sha256(
        f"{CLUSTER_VERSION}|{snapshot_hash}".encode()
    ).hexdigest()[:20]
    revisions, groups = cluster_rows(rows)
    event_assignments: list[tuple[str, str, str, datetime]] = []
    asset_links: dict[tuple[str, str], tuple[datetime, str]] = {}
    catalyst_rows = []

    for group in groups:
        cid = _catalyst_id(revisions, group, build_id)
        members = [revisions[i] for i in group]
        earliest = min(members, key=lambda r: (r.public_time, r.revision_id))
        for revision in members:
            event_assignments.extend(
                (cid, build_id, event_id, revision.public_time)
                for event_id in revision.event_ids
            )
            for ticker, event_id, public_time in revision.ticker_links:
                key = (cid, ticker)
                if key not in asset_links or public_time < asset_links[key][0]:
                    asset_links[key] = (public_time, event_id)
        n_assets = sum(1 for catalyst_id, _ in asset_links if catalyst_id == cid)
        catalyst_rows.append((
            cid, build_id, CLUSTER_VERSION, earliest.public_time, earliest.headline[:500],
            len(members), n_assets, datetime.now(timezone.utc).replace(tzinfo=None),
            json.dumps({
                "window_hours": CLUSTER_WINDOW.total_seconds() / 3600,
                "token_jaccard_min": TOKEN_JACCARD_MIN,
                "sequence_similarity_min": SEQUENCE_SIM_MIN,
                "requires_entity_overlap": True,
                "maximum_total_cluster_span_hours":
                    CLUSTER_WINDOW.total_seconds() / 3600,
            }),
        ))

    catalyst_frame = pd.DataFrame(catalyst_rows, columns=[
        "catalyst_id", "build_id", "cluster_version", "earliest_public_time", "headline",
        "n_article_revisions", "n_assets", "created_at", "metadata",
    ])
    event_frame = pd.DataFrame(event_assignments, columns=[
        "catalyst_id", "build_id", "event_id", "event_public_time",
    ])
    asset_frame = pd.DataFrame(
        [(cid, build_id, ticker, public_time, event_id)
         for (cid, ticker), (public_time, event_id) in asset_links.items()],
        columns=["catalyst_id", "build_id", "ticker", "first_link_public_time",
                 "source_event_id"],
    )
    # DuckDB's row-wise executemany is prohibitively slow for tens of thousands
    # of news rows. Registered frames keep the same transaction semantics while
    # using vectorized INSERT ... SELECT operations.
    con.register("_new_catalysts", catalyst_frame)
    con.register("_new_catalyst_events", event_frame)
    con.register("_new_catalyst_assets", asset_frame)
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute("""
            INSERT INTO catalyst_builds
                (build_id,cluster_version,data_snapshot_hash,created_at,status,metadata)
            VALUES (?,?,?,CURRENT_TIMESTAMP,'BUILDING',?) ON CONFLICT DO NOTHING
        """, [build_id, CLUSTER_VERSION, snapshot_hash,
              json.dumps({"event_rows": len(rows), "immutable_snapshot": True})])
        # Append provenance first. Re-running a version never deletes or replaces
        # an earlier catalyst assignment.
        con.execute("""
            INSERT INTO catalysts
            (catalyst_id, build_id, cluster_version, earliest_public_time, headline,
             n_article_revisions, n_assets, created_at, metadata)
            SELECT catalyst_id, build_id, cluster_version, earliest_public_time, headline,
                   n_article_revisions, n_assets, created_at, metadata
            FROM _new_catalysts ON CONFLICT DO NOTHING
        """)
        con.execute("""
            INSERT INTO catalyst_events
            (catalyst_id, build_id, event_id, event_public_time)
            SELECT catalyst_id, build_id, event_id, event_public_time
            FROM _new_catalyst_events ON CONFLICT DO NOTHING
        """)
        con.execute("""
            INSERT INTO catalyst_assets
            (catalyst_id, build_id, ticker, first_link_public_time, source_event_id)
            SELECT catalyst_id, build_id, ticker, first_link_public_time, source_event_id
            FROM _new_catalyst_assets ON CONFLICT DO NOTHING
        """)

        # This pointer may advance to a newer clustering algorithm; the normalized
        # membership tables above retain all historical versions.
        con.execute("""
            UPDATE events
            SET catalyst_id = _new_catalyst_events.catalyst_id
            FROM _new_catalyst_events
            WHERE events.event_id = _new_catalyst_events.event_id
        """)
        con.execute("""
            UPDATE catalyst_builds SET status='COMPLETE'
            WHERE build_id=?
        """, [build_id])
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        con.close()
        raise

    con.close()
    stats = {
        "raw_ticker_event_rows": len(rows),
        "raw_articles": len({revision.identity for revision in revisions}),
        "unique_article_revisions": len(revisions),
        "catalysts": len(groups),
        "catalyst_ticker_observations": len(asset_links),
        "cluster_version": CLUSTER_VERSION,
        "build_id": build_id,
        "data_snapshot_hash": snapshot_hash,
    }
    if verbose:
        print(
            f"Catalysts ({CLUSTER_VERSION}): ticker-event rows={stats['raw_ticker_event_rows']}  "
            f"raw articles={stats['raw_articles']}  revisions={stats['unique_article_revisions']}  "
            f"catalysts={stats['catalysts']}  "
            f"catalyst-ticker obs={stats['catalyst_ticker_observations']}"
        )
        print("  No ticker-row/catalyst ratio is reported as a duplication rate.")
    return stats


if __name__ == "__main__":
    build()
