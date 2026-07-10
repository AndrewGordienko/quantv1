"""Time-valid actor registry and explicitly non-causal news-mention extraction.

Actor identity, aliases, institutional roles and asset exposure are normalized
and independently time-valid.  Registry writes are additive/idempotent; no
table is truncated.  Headline alias matches are always classified as
``merely_mentioned`` and are ineligible for the primary actor hypothesis.

Primary B2 events must come from a primary source with a resolved participation
role such as ``speaker_author``, ``direct_public_action`` or
``verified_decision_maker``.  This module intentionally does not promote a
headline regex match into any of those causal roles.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import re

from ..db import connect

REGISTRY_VERSION = "actor-registry-v3-sourced-temporal"
EXTRACTION_VERSION = "news-context-v3-temporal-alias"
PRIMARY_EVENT_ROLES = {
    "speaker_author", "direct_public_action", "verified_decision_maker",
}
ACTOR_EVENT_ROLES = PRIMARY_EVENT_ROLES | {
    "directly_quoted", "meeting_participant", "subject_of_story",
    "merely_mentioned",
}


@dataclass(frozen=True)
class Actor:
    actor_id: str
    name: str
    actor_type: str
    source: str
    legacy_authority_prior: float


@dataclass(frozen=True)
class Role:
    actor_id: str
    organization: str
    role: str
    valid_from: str
    valid_to: str | None
    source: str


@dataclass(frozen=True)
class Alias:
    actor_id: str
    alias: str
    valid_from: str
    valid_to: str | None
    source: str
    entity_link_required: bool = False


@dataclass(frozen=True)
class Exposure:
    actor_id: str
    ticker: str
    valid_from: str
    valid_to: str | None
    channel: str
    confidence: float
    source: str


FED_POWELL = "https://www.federalreserve.gov/aboutthefed/bios/board/powell.htm"
FED_WARSH = "https://www.federalreserve.gov/aboutthefed/bios/board/warsh.htm"
SEC_GENSLER = "https://www.sec.gov/about/sec-commissioners/gary-gensler"
SEC_ATKINS = "https://www.sec.gov/newsroom/press-releases/2025-68"
SEC_UYEDA = "https://www.sec.gov/newsroom/press-releases/2025-29"
TREASURY_BESSENT = "https://home.treasury.gov/about/general-information/officials/scott-bessent"
WH_HASSETT = "https://www.whitehouse.gov/wp-content/uploads/2025/03/M-25-11-Guidance-Unleashing-American-Energy.pdf"

ACTORS = (
    Actor("jerome_powell", "Jerome H. Powell", "central_banker", FED_POWELL, 0.95),
    Actor("kevin_warsh", "Kevin M. Warsh", "central_banker", FED_WARSH, 0.95),
    Actor("elon_musk", "Elon Musk", "executive", "https://ir.tesla.com/corporate/elon-musk", 0.90),
    Actor("jensen_huang", "Jensen Huang", "executive", "https://www.nvidia.com/en-us/about-nvidia/board-of-directors/jensen-huang/", 0.80),
    Actor("tim_cook", "Tim Cook", "executive", "https://www.apple.com/uk/newsroom/2011/08/24Steve-Jobs-Resigns-as-CEO-of-Apple/", 0.70),
    Actor("satya_nadella", "Satya Nadella", "executive", "https://news.microsoft.com/source/2014/02/04/microsoft-board-names-satya-nadella-as-ceo/", 0.70),
    Actor("andy_jassy", "Andy Jassy", "executive", "https://ir.aboutamazon.com/officers-and-directors/person-details/default.aspx?ItemId=be0a9875-4456-4f84-b4d9-ec2e4b97d1d8", 0.60),
    Actor("sundar_pichai", "Sundar Pichai", "executive", "https://abc.xyz/investor/news/news-details/2019/Alphabet-management-change-12-03-2019/default.aspx", 0.70),
    Actor("mark_zuckerberg", "Mark Zuckerberg", "executive", "https://about.fb.com/news/tag/mark-zuckerberg/", 0.75),
    Actor("jamie_dimon", "Jamie Dimon", "executive", "https://www.jpmorganchase.com/about/leadership/jamie-dimon", 0.70),
    Actor("lisa_su", "Lisa Su", "executive", "https://ir.amd.com/news-events/press-releases/detail/568/amd-appoints-dr-lisa-su-as-president-and-chief-executive-officer", 0.65),
    Actor("kevin_hassett", "Kevin Hassett", "government_official", WH_HASSETT, 0.60),
    Actor("scott_bessent", "Scott Bessent", "government_official", TREASURY_BESSENT, 0.70),
    Actor("gary_gensler", "Gary Gensler", "regulator", SEC_GENSLER, 0.60),
    Actor("mark_uyeda", "Mark T. Uyeda", "regulator", SEC_UYEDA, 0.60),
    Actor("paul_atkins", "Paul S. Atkins", "regulator", SEC_ATKINS, 0.60),
)

ROLES = (
    Role("jerome_powell", "Federal Reserve Board", "Governor", "2012-05-25", "2028-01-31", FED_POWELL),
    Role("jerome_powell", "Federal Reserve Board", "Chair", "2018-02-05", "2026-05-22", FED_POWELL),
    Role("kevin_warsh", "Federal Reserve Board", "Chair", "2026-05-22", "2030-05-21", FED_WARSH),
    Role("elon_musk", "Tesla", "Chief Executive Officer", "2008-10-31", None, ACTORS[2].source),
    Role("jensen_huang", "NVIDIA", "President and Chief Executive Officer", "1993-12-31", None, ACTORS[3].source),
    Role("tim_cook", "Apple", "Chief Executive Officer", "2011-08-24", None, ACTORS[4].source),
    Role("satya_nadella", "Microsoft", "Chief Executive Officer", "2014-02-04", None, ACTORS[5].source),
    Role("andy_jassy", "Amazon", "Chief Executive Officer", "2021-07-05", None, ACTORS[6].source),
    Role("sundar_pichai", "Alphabet", "Chief Executive Officer", "2019-12-03", None, ACTORS[7].source),
    # Conservative sourced-as-of date: Meta's official page identifies him as CEO.
    Role("mark_zuckerberg", "Meta", "Chief Executive Officer", "2009-02-26", None, "https://about.fb.com/news/2009/02/facebook-opens-governance-of-service-and-policy-process-to-users/"),
    Role("jamie_dimon", "JPMorgan Chase", "Chief Executive Officer", "2006-01-01", None, ACTORS[9].source),
    Role("lisa_su", "AMD", "President and Chief Executive Officer", "2014-10-08", None, ACTORS[10].source),
    Role("kevin_hassett", "National Economic Council", "Director", "2025-01-21", None, WH_HASSETT),
    Role("scott_bessent", "U.S. Department of the Treasury", "Secretary", "2025-01-28", None, TREASURY_BESSENT),
    Role("gary_gensler", "U.S. Securities and Exchange Commission", "Chair", "2021-04-17", "2025-01-20", SEC_GENSLER),
    Role("mark_uyeda", "U.S. Securities and Exchange Commission", "Acting Chair", "2025-01-21", "2025-04-21", SEC_UYEDA),
    Role("paul_atkins", "U.S. Securities and Exchange Commission", "Chair", "2025-04-21", None, SEC_ATKINS),
)


def _aliases() -> tuple[Alias, ...]:
    aliases: list[Alias] = []
    role_by_actor = {role.actor_id: role for role in ROLES}
    for actor in ACTORS:
        role = role_by_actor[actor.actor_id]
        aliases.append(Alias(actor.actor_id, actor.name, role.valid_from,
                             role.valid_to, actor.source))
    aliases.extend([
        Alias("jerome_powell", "Jerome Powell", "2012-05-25", "2028-01-31", FED_POWELL),
        Alias("jerome_powell", "Powell", "2012-05-25", "2028-01-31", FED_POWELL),
        Alias("jerome_powell", "Fed Chair", "2018-02-05", "2026-05-22", FED_POWELL, True),
        Alias("kevin_warsh", "Kevin Warsh", "2026-05-22", "2030-05-21", FED_WARSH),
        Alias("kevin_warsh", "Fed Chair", "2026-05-22", "2030-05-21", FED_WARSH, True),
        Alias("elon_musk", "Musk", "2008-10-31", None, ACTORS[2].source),
        # Surnames such as "Huang" and occupational labels are not safe aliases.
        Alias("jensen_huang", "Jensen Huang", "1993-12-31", None, ACTORS[3].source),
        Alias("tim_cook", "Tim Cook", "2011-08-24", None, ACTORS[4].source),
        Alias("satya_nadella", "Nadella", "2014-02-04", None, ACTORS[5].source),
        Alias("andy_jassy", "Jassy", "2021-07-05", None, ACTORS[6].source),
        Alias("sundar_pichai", "Pichai", "2019-12-03", None, ACTORS[7].source),
        Alias("mark_zuckerberg", "Zuckerberg", "2009-02-26", None, ACTORS[8].source),
        Alias("jamie_dimon", "Dimon", "2006-01-01", None, ACTORS[9].source),
        Alias("lisa_su", "Lisa Su", "2014-10-08", None, ACTORS[10].source),
        Alias("kevin_hassett", "Hassett", "2025-01-21", None, WH_HASSETT),
        Alias("scott_bessent", "Bessent", "2025-01-28", None, TREASURY_BESSENT),
        Alias("gary_gensler", "Gensler", "2021-04-17", "2025-01-20", SEC_GENSLER),
        Alias("gary_gensler", "SEC Chair", "2021-04-17", "2025-01-20", SEC_GENSLER, True),
        Alias("mark_uyeda", "Uyeda", "2025-01-21", None, SEC_UYEDA),
        Alias("mark_uyeda", "SEC Chair", "2025-01-21", "2025-04-21", SEC_UYEDA, True),
        Alias("paul_atkins", "Paul Atkins", "2025-04-21", None, SEC_ATKINS),
        Alias("paul_atkins", "Atkins", "2025-04-21", None, SEC_ATKINS),
        Alias("paul_atkins", "SEC Chair", "2025-04-21", None, SEC_ATKINS, True),
    ])
    return tuple({(a.actor_id, a.alias.lower(), a.valid_from): a for a in aliases}.values())


ALIASES = _aliases()

_CORPORATE_EXPOSURES = {
    "elon_musk": "TSLA", "jensen_huang": "NVDA", "tim_cook": "AAPL",
    "satya_nadella": "MSFT", "andy_jassy": "AMZN", "sundar_pichai": "GOOGL",
    "mark_zuckerberg": "META", "jamie_dimon": "JPM", "lisa_su": "AMD",
}


def _exposures() -> tuple[Exposure, ...]:
    role_by_actor = {role.actor_id: role for role in ROLES}
    rows = [
        Exposure(actor_id, ticker, role_by_actor[actor_id].valid_from,
                 role_by_actor[actor_id].valid_to, "executive_control", 1.0,
                 role_by_actor[actor_id].source)
        for actor_id, ticker in _CORPORATE_EXPOSURES.items()
    ]
    for actor_id, start, end, source in (
        ("jerome_powell", "2018-02-05", "2026-05-22", FED_POWELL),
        ("kevin_warsh", "2026-05-22", "2030-05-21", FED_WARSH),
    ):
        rows.extend(Exposure(actor_id, ticker, start, end, "monetary_policy", confidence, source)
                    for ticker, confidence in (("TLT", 1.0), ("IEF", 1.0), ("XLF", 0.8), ("SPY", 0.6)))
    for actor_id, start, end, source in (
        ("gary_gensler", "2021-04-17", "2025-01-20", SEC_GENSLER),
        ("mark_uyeda", "2025-01-21", "2025-04-21", SEC_UYEDA),
        ("paul_atkins", "2025-04-21", None, SEC_ATKINS),
    ):
        rows.extend([
            Exposure(actor_id, "XLF", start, end, "securities_regulation", 0.8, source),
            Exposure(actor_id, "SPY", start, end, "securities_regulation", 0.5, source),
        ])
    rows.extend([
        Exposure("kevin_hassett", "SPY", "2025-01-21", None, "economic_policy", 0.6, WH_HASSETT),
        Exposure("scott_bessent", "TLT", "2025-01-28", None, "treasury_policy", 0.9, TREASURY_BESSENT),
        Exposure("scott_bessent", "SPY", "2025-01-28", None, "treasury_policy", 0.6, TREASURY_BESSENT),
    ])
    return tuple(rows)


EXPOSURES = _exposures()


def _event_id(actor_id: str, source_event_id: str, ticker: str) -> str:
    value = f"{EXTRACTION_VERSION}|{actor_id}|{source_event_id}|{ticker}"
    return hashlib.sha1(value.encode()).hexdigest()[:20]


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def _valid_on(when: date, valid_from, valid_to) -> bool:
    start = _as_date(valid_from)
    end = _as_date(valid_to) if valid_to else None
    return start <= when and (end is None or when <= end)


def register(verbose: bool = True) -> dict:
    """Insert sourced registry records without truncating historical tables."""
    con = connect()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for actor in ACTORS:
        metadata = json.dumps({
            "legacy_authority_prior": actor.legacy_authority_prior,
            "predictive_feature_allowed": False,
            "note": "hand-assigned prior retained as non-model metadata only",
        })
        # Correct stable canonical identity fields; temporal facts remain in the
        # append-only child tables below.
        con.execute("""
            INSERT INTO actors
                (actor_id, name, actor_type, metadata, registry_version,
                 registry_status, source, first_seen_at)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT (actor_id) DO UPDATE SET
                name=excluded.name, actor_type=excluded.actor_type,
                metadata=excluded.metadata, registry_version=excluded.registry_version,
                registry_status=excluded.registry_status, source=excluded.source
        """, [actor.actor_id, actor.name, actor.actor_type, metadata,
              REGISTRY_VERSION, "ACTIVE", actor.source, now])
    # Old databases carried denormalized role/org/alias and authority columns.
    # Their content is unsafe for point-in-time joins.  The prior is preserved in
    # metadata above; neutralize the deprecated feature columns if they exist.
    legacy_columns = {row[1] for row in con.execute("PRAGMA table_info('actors')").fetchall()}
    unsafe = [column for column in ("role", "org", "authority", "aliases")
              if column in legacy_columns]
    if unsafe:
        assignments = ", ".join(f"{column}=NULL" for column in unsafe)
        con.execute(f"UPDATE actors SET {assignments}")
    con.executemany("""
        INSERT INTO actor_aliases
            (actor_id, alias, valid_from, valid_to, source, record_version,
             entity_link_required, first_seen_at)
        VALUES (?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
    """, [(a.actor_id, a.alias, a.valid_from, a.valid_to, a.source,
            REGISTRY_VERSION, a.entity_link_required, now) for a in ALIASES])
    con.executemany("""
        INSERT INTO actor_roles
            (actor_id, organization, role, valid_from, valid_to, source,
             record_version, first_seen_at)
        VALUES (?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
    """, [(r.actor_id, r.organization, r.role, r.valid_from, r.valid_to,
            r.source, REGISTRY_VERSION, now) for r in ROLES])
    con.executemany("""
        INSERT INTO actor_asset_exposure
            (actor_id, ticker, valid_from, valid_to, channel, confidence, source,
             record_version, first_seen_at)
        VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
    """, [(x.actor_id, x.ticker, x.valid_from, x.valid_to, x.channel,
            x.confidence, x.source, REGISTRY_VERSION, now) for x in EXPOSURES])
    counts = {
        "actors": con.execute("SELECT COUNT(*) FROM actors WHERE registry_version=?",
                              [REGISTRY_VERSION]).fetchone()[0],
        "aliases": con.execute("SELECT COUNT(*) FROM actor_aliases WHERE record_version=?",
                               [REGISTRY_VERSION]).fetchone()[0],
        "roles": con.execute("SELECT COUNT(*) FROM actor_roles WHERE record_version=?",
                             [REGISTRY_VERSION]).fetchone()[0],
        "exposures": con.execute("SELECT COUNT(*) FROM actor_asset_exposure WHERE record_version=?",
                                 [REGISTRY_VERSION]).fetchone()[0],
    }
    con.close()
    if verbose:
        print(f"Actor registry {REGISTRY_VERSION}: {counts}")
        print("  Authority priors are metadata only and forbidden as model features.")
    return counts


def extract_from_news(verbose: bool = True) -> dict:
    """Create contextual mention rows; never treat them as actor actions."""
    register(verbose=False)
    con = connect()
    aliases = con.execute("""
        SELECT actor_id, alias, valid_from, valid_to, entity_link_required
        FROM actor_aliases WHERE record_version=?
    """, [REGISTRY_VERSION]).fetchall()
    patterns = [(actor_id, alias, valid_from, valid_to, needs_link,
                 re.compile(r"\b" + re.escape(alias) + r"\b", re.I))
                for actor_id, alias, valid_from, valid_to, needs_link in aliases]
    rows = con.execute("""
        SELECT event_id, ticker, source_time, catalyst_id, source_url, payload
        FROM events
        WHERE layer='N' AND ticker IS NOT NULL AND source_time IS NOT NULL
    """).fetchall()

    output = []
    seen: set[tuple[str, str]] = set()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for source_event_id, ticker, public_time, catalyst_id, source_url, raw_payload in rows:
        try:
            payload = json.loads(raw_payload or "{}")
        except (TypeError, ValueError):
            continue
        headline = str(payload.get("title") or payload.get("headline") or "")
        when = _as_date(public_time)
        matches: dict[str, list[tuple[str, bool]]] = {}
        for actor_id, alias, valid_from, valid_to, needs_link, pattern in patterns:
            if _valid_on(when, valid_from, valid_to) and pattern.search(headline):
                matches.setdefault(actor_id, []).append((alias, needs_link))

        # A role-title alias is usable only if temporal resolution leaves one
        # possible person. Full names/safe surnames remain direct alias matches.
        role_alias_owners: dict[str, set[str]] = {}
        for actor_id, actor_matches in matches.items():
            for alias, needs_link in actor_matches:
                if needs_link:
                    role_alias_owners.setdefault(alias.lower(), set()).add(actor_id)
        for actor_id, actor_matches in matches.items():
            usable = [alias for alias, needs_link in actor_matches
                      if not needs_link or len(role_alias_owners[alias.lower()]) == 1]
            if not usable:
                continue
            actor_event_id = _event_id(actor_id, source_event_id, ticker)
            if (actor_event_id, ticker) in seen:
                continue
            seen.add((actor_event_id, ticker))
            evidence = f"time-valid alias match: {sorted(usable, key=len)[-1]}"
            output.append((
                actor_event_id, actor_id, ticker, public_time, "news_context",
                headline[:500], catalyst_id, source_url or "news_event", now,
                source_event_id, "merely_mentioned", 0.6, evidence, False,
                EXTRACTION_VERSION,
                json.dumps({"primary_source": False,
                            "entity_link_method": "temporal_alias_resolution"}),
            ))
    con.executemany("""
        INSERT INTO actor_events
            (actor_event_id, actor_id, ticker, public_time, event_type, headline,
             catalyst_id, source, first_seen_at, source_event_id,
             actor_event_role, role_confidence, role_evidence,
             primary_hypothesis_eligible, extraction_version, metadata)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
    """, output)
    by_actor = dict(con.execute("""
        SELECT actor_id, COUNT(*) FROM actor_events
        WHERE extraction_version=? GROUP BY 1 ORDER BY 2 DESC
    """, [EXTRACTION_VERSION]).fetchall())
    con.close()
    if verbose:
        print(f"Context-only actor mentions ({EXTRACTION_VERSION}): {len(output)} candidates")
        print("  All are role=merely_mentioned and primary_hypothesis_eligible=false.")
    return {
        "actor_events": len(output), "by_actor": by_actor,
        "study_eligibility": "CONTEXT_ONLY_NOT_B2",
    }


if __name__ == "__main__":
    register()
    extract_from_news()
