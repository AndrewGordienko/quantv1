"""Strict primary-source ingestion for the Fed B2 and B3 pilots.

The input is a reviewed JSON/JSONL manifest because exact public and segment
timestamps cannot be reconstructed safely from page publication dates.  Source
URLs must be official Federal Reserve Board or Reserve Bank domains.

Two samples are intentionally distinct:

* ``B2_FED_SPEAKER_PANEL`` — speeches/remarks from Chairs, Governors, regional
  Reserve Bank presidents and other voting participants.
* ``B3_CHAIR_PRESS_CONFERENCE`` — timestamped prepared/question/answer segments
  for within-conference state changes.

See ``docs/fed_primary_manifest.md`` for the manifest contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from ..db import connect

B2_SAMPLE = "B2_FED_SPEAKER_PANEL"
B3_SAMPLE = "B3_CHAIR_PRESS_CONFERENCE"
SAMPLES = {B2_SAMPLE, B3_SAMPLE}
OFFICIAL_DOMAINS = {
    "federalreserve.gov", "www.federalreserve.gov",
    "bostonfed.org", "www.bostonfed.org",
    "newyorkfed.org", "www.newyorkfed.org",
    "philadelphiafed.org", "www.philadelphiafed.org",
    "clevelandfed.org", "www.clevelandfed.org",
    "richmondfed.org", "www.richmondfed.org",
    "atlantafed.org", "www.atlantafed.org",
    "chicagofed.org", "www.chicagofed.org",
    "stlouisfed.org", "www.stlouisfed.org",
    "minneapolisfed.org", "www.minneapolisfed.org",
    "kansascityfed.org", "www.kansascityfed.org",
    "dallasfed.org", "www.dallasfed.org",
    "frbsf.org", "www.frbsf.org",
}
PRIMARY_ROLES = {"speaker_author", "direct_public_action", "verified_decision_maker"}
B2_TYPES = {"speech", "remarks", "testimony", "public_interview", "policy_statement"}
B3_SEGMENT_ROLES = {"prepared", "question", "answer"}


class ManifestError(ValueError):
    """Raised when a record would weaken point-in-time or source provenance."""


def _load(path: str | Path) -> list[dict]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data.get("communications", [])


def _official_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in OFFICIAL_DOMAINS


def _utc_naive(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ManifestError(f"{field} must include an explicit UTC offset")
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _date(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{field} must be YYYY-MM-DD") from exc


def _communication_id(record: dict) -> str:
    supplied = record.get("communication_id")
    if supplied:
        return str(supplied)
    raw = f"{record['sample']}|{record['actor']['actor_id']}|{record['public_time']}|{record['source_url']}"
    return hashlib.sha1(raw.encode()).hexdigest()[:24]


def _validate(record: dict) -> dict:
    sample = record.get("sample")
    if sample not in SAMPLES:
        raise ManifestError(f"sample must be one of {sorted(SAMPLES)}")
    actor = record.get("actor") or {}
    for field in ("actor_id", "name", "actor_type"):
        if not actor.get(field):
            raise ManifestError(f"actor.{field} is required")
    if not _official_url(record.get("source_url", "")):
        raise ManifestError("source_url must be an official Federal Reserve HTTPS URL")
    public_time = _utc_naive(record.get("public_time"), "public_time")
    if record.get("timestamp_precision") != "exact":
        raise ManifestError("timestamp_precision must be 'exact'; page dates are insufficient")
    event_role = record.get("actor_event_role")
    if event_role not in PRIMARY_ROLES:
        raise ManifestError(f"actor_event_role must be one of {sorted(PRIMARY_ROLES)}")
    communication_type = record.get("communication_type")
    if sample == B2_SAMPLE and communication_type not in B2_TYPES:
        raise ManifestError(f"B2 communication_type must be one of {sorted(B2_TYPES)}")
    if sample == B3_SAMPLE and communication_type != "chair_press_conference":
        raise ManifestError("B3 communication_type must be chair_press_conference")

    institutional_role = record.get("institutional_role") or {}
    for field in ("organization", "role", "valid_from", "source"):
        if not institutional_role.get(field):
            raise ManifestError(f"institutional_role.{field} is required")
    _date(institutional_role["valid_from"], "institutional_role.valid_from")
    _date(institutional_role.get("valid_to"), "institutional_role.valid_to")
    if not _official_url(institutional_role["source"]):
        raise ManifestError("institutional_role.source must be official")

    exposures = record.get("asset_exposures") or []
    if not exposures:
        raise ManifestError("at least one rates/financial asset_exposure is required")
    for exposure in exposures:
        if not exposure.get("ticker") or not exposure.get("channel"):
            raise ManifestError("every asset_exposure needs ticker and channel")
        if not 0 <= float(exposure.get("confidence", -1)) <= 1:
            raise ManifestError("asset_exposure confidence must be in [0,1]")
        if not _official_url(exposure.get("source", record["source_url"])):
            raise ManifestError("asset_exposure source must be official")
    tickers = {str(exposure["ticker"]).upper() for exposure in exposures}
    if not ({"IEF", "TLT"} & tickers) or "XLF" not in tickers:
        raise ManifestError("pilot requires Treasury-duration exposure (IEF/TLT) and XLF")

    segments = record.get("segments") or []
    if sample == B3_SAMPLE:
        if not segments:
            raise ManifestError("B3 requires timestamped transcript/audio segments")
        previous = None
        for index, segment in enumerate(segments):
            if segment.get("segment_role") not in B3_SEGMENT_ROLES:
                raise ManifestError(f"segments[{index}].segment_role is invalid")
            timestamp = _utc_naive(segment.get("public_time"),
                                   f"segments[{index}].public_time")
            if timestamp < public_time or (previous and timestamp < previous):
                raise ManifestError("B3 segment timestamps must be monotone and post-release")
            previous = timestamp
            if not str(segment.get("text") or "").strip():
                raise ManifestError(f"segments[{index}].text is required")
    elif segments:
        raise ManifestError("B2 panel records must not contain B3 conference segments")
    return {**record, "_public_time": public_time,
            "_communication_id": _communication_id(record)}


def ingest_manifest(path: str | Path, verbose: bool = True) -> dict:
    """Validate and append a reviewed primary-source manifest."""
    validated = [_validate(record) for record in _load(path)]
    con = connect()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    communication_count = segment_count = event_count = 0
    con.execute("BEGIN TRANSACTION")
    try:
        for record in validated:
            actor = record["actor"]
            communication_id = record["_communication_id"]
            role = record["institutional_role"]
            metadata = {
                "sample": record["sample"],
                "timestamp_precision": "exact",
                "primary_source": True,
                "audio_url": record.get("audio_url"),
                "topics": record.get("topics", []),
            }
            con.execute("""
                INSERT INTO actors
                    (actor_id, name, actor_type, metadata, registry_version,
                     registry_status, source, first_seen_at)
                VALUES (?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
            """, [actor["actor_id"], actor["name"], actor["actor_type"],
                  json.dumps({"predictive_feature_allowed": False}),
                  "fed-primary-v1", "ACTIVE", role["source"], now])
            con.execute("""
                INSERT INTO actor_aliases
                    (actor_id, alias, valid_from, valid_to, source, record_version,
                     entity_link_required, first_seen_at)
                VALUES (?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
            """, [actor["actor_id"], actor["name"], role["valid_from"],
                  role.get("valid_to"), role["source"], "fed-primary-v1", False, now])
            con.execute("""
                INSERT INTO actor_roles
                    (actor_id, organization, role, valid_from, valid_to, source,
                     record_version, first_seen_at)
                VALUES (?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
            """, [actor["actor_id"], role["organization"], role["role"],
                  role["valid_from"], role.get("valid_to"), role["source"],
                  "fed-primary-v1", now])
            transcript = str(record.get("transcript") or "")
            source_hash = hashlib.sha256(transcript.encode()).hexdigest() if transcript else None
            con.execute("""
                INSERT INTO fed_communications
                    (communication_id, actor_id, public_time, communication_type,
                     title, source_url, transcript, prepared_or_qa, source_sha256,
                     first_seen_at, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
            """, [communication_id, actor["actor_id"], record["_public_time"],
                  record["communication_type"], record.get("title"),
                  record["source_url"], transcript,
                  "mixed" if record["sample"] == B3_SAMPLE else "prepared",
                  source_hash, now, json.dumps(metadata)])
            communication_count += 1

            for exposure in record["asset_exposures"]:
                ticker = str(exposure["ticker"]).upper()
                exposure_source = exposure.get("source", record["source_url"])
                con.execute("""
                    INSERT INTO actor_asset_exposure
                        (actor_id, ticker, valid_from, valid_to, channel, confidence,
                         source, record_version, first_seen_at)
                    VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
                """, [actor["actor_id"], ticker, role["valid_from"],
                      role.get("valid_to"), exposure["channel"],
                      float(exposure["confidence"]), exposure_source,
                      "fed-primary-v1", now])
                actor_event_id = hashlib.sha1(
                    f"fed-primary-v1|{communication_id}|{ticker}".encode()
                ).hexdigest()[:20]
                event_metadata = {
                    **metadata,
                    "actor_asset_channel": exposure["channel"],
                    "actor_asset_exposure_confidence": float(exposure["confidence"]),
                }
                con.execute("""
                    INSERT INTO actor_events
                        (actor_event_id, actor_id, ticker, public_time, event_type,
                         headline, catalyst_id, source, first_seen_at,
                         source_event_id, actor_event_role, role_confidence,
                         role_evidence, primary_hypothesis_eligible,
                         extraction_version, metadata)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
                """, [actor_event_id, actor["actor_id"], ticker,
                      record["_public_time"], record["communication_type"],
                      record.get("title"), communication_id, record["source_url"],
                      now, communication_id, record["actor_event_role"], 1.0,
                      "official primary-source manifest", True,
                      "fed-primary-v1", json.dumps(event_metadata)])
                event_count += 1

            for index, segment in enumerate(record.get("segments") or []):
                segment_time = _utc_naive(segment["public_time"],
                                          f"segments[{index}].public_time")
                segment_id = hashlib.sha1(
                    f"{communication_id}|{index}|{segment_time.isoformat()}|{segment['text']}".encode()
                ).hexdigest()[:24]
                con.execute("""
                    INSERT INTO fed_transcript_segments
                        (segment_id, communication_id, segment_index,
                         segment_public_time, actor_id, segment_role, text,
                         source_url, first_seen_at, metadata)
                    VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING
                """, [segment_id, communication_id, index, segment_time,
                      segment.get("actor_id"), segment["segment_role"],
                      segment["text"], record["source_url"], now,
                      json.dumps({"sample": B3_SAMPLE,
                                  "audio_offset_seconds": segment.get("audio_offset_seconds")})])
                segment_count += 1
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        con.close()
        raise
    con.close()
    result = {
        "communications": communication_count,
        "actor_asset_events": event_count,
        "segments": segment_count,
        "b2_communications": sum(r["sample"] == B2_SAMPLE for r in validated),
        "b3_press_conferences": sum(r["sample"] == B3_SAMPLE for r in validated),
    }
    if verbose:
        print(f"Fed primary-source ingest: {result}")
    return result
