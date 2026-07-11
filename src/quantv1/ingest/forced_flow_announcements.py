"""Verified announcement-timestamp corpus for the S&P 500 addition census.

This unblocks the announcement->effective continuation test. It is deliberately
BLIND to returns: it never imports prices or computes an outcome. The worklist
fixes the denominator (all 113 addition batches) before searching; resolved
records are validated against source tiers and frozen into a manifest + a
rejection ledger before anything is joined to market outcomes.

See goldset/forced_flow/announcement_rules.json for the frozen rules.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json

from .forced_flow import CENSUS_VERSION
from ..db import connect

RULES_VERSION = "forced-flow-announcement-rules-v1"
GOLDSET_DIR = "goldset/forced_flow"
WORKLIST_PATH = f"{GOLDSET_DIR}/announcement_worklist_v1.jsonl"
MANIFEST_PATH = f"{GOLDSET_DIR}/announcement_manifest_v1.jsonl"
LEDGER_PATH = f"{GOLDSET_DIR}/announcement_rejection_ledger_v1.json"

VALID_TIERS = {1, 2, 3}
VALID_PRECISION = {"exact_minute", "exact_hour"}
VALID_KIND = {"ORIGINAL", "CORRECTION"}
# US regular session in UTC: 13:30-20:00 (used only to label entry timing).
_SESSION_OPEN_MIN = 13 * 60 + 30
_SESSION_CLOSE_MIN = 20 * 60


class AnnouncementRejection(ValueError):
    """Raised when a candidate record fails a frozen tier/precision rule."""

    def __init__(self, reason_code: str, detail: str = ""):
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


def _addition_batches(con) -> dict[str, dict]:
    rows = con.execute("""
        SELECT event_batch_id, effective_date, ticker, coverage_status
        FROM forced_flow_events
        WHERE version=? AND event_type='addition'
    """, [CENSUS_VERSION]).fetchall()
    batches: dict[str, dict] = {}
    for batch_id, effective, ticker, coverage in rows:
        entry = batches.setdefault(batch_id, {
            "effective_date": effective.isoformat(), "tickers": [],
            "covered_tickers": []})
        entry["tickers"].append(ticker)
        if coverage == "COVERED":
            entry["covered_tickers"].append(ticker)
    return batches


def generate_worklist(verbose: bool = True) -> dict:
    """Freeze the denominator: every addition batch, UNRESOLVED. No returns."""
    con = connect(read_only=True)
    batches = _addition_batches(con)
    con.close()
    lines = []
    for batch_id in sorted(batches, key=lambda b: batches[b]["effective_date"]):
        info = batches[batch_id]
        lines.append({
            "event_batch_id": batch_id,
            "effective_date": info["effective_date"],
            "affected_tickers": sorted(info["tickers"]),
            "covered_tickers": sorted(info["covered_tickers"]),
            "verification_status": "UNRESOLVED",
        })
    with open(WORKLIST_PATH, "w") as file:
        for line in lines:
            file.write(json.dumps(line) + "\n")
    summary = {"rules_version": RULES_VERSION, "batches": len(lines),
               "path": WORKLIST_PATH}
    if verbose:
        print(f"announcement worklist: {len(lines)} addition batches "
              f"enumerated (all UNRESOLVED) -> {WORKLIST_PATH}")
    return summary


def classify_entry(public_time: datetime) -> str:
    """Executable entry is the next session open (daily bars only for adds)."""
    utc = public_time.astimezone(timezone.utc)
    minute_of_day = utc.hour * 60 + utc.minute
    weekday = utc.weekday() < 5
    if weekday and _SESSION_OPEN_MIN <= minute_of_day < _SESSION_CLOSE_MIN:
        return "INTRADAY_NEXT_SESSION_OPEN"
    return "AFTER_HOURS_NEXT_SESSION_OPEN"


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise AnnouncementRejection("DATE_ONLY",
                                    "announcement_public_time not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise AnnouncementRejection("DATE_ONLY",
                                    "announcement_public_time has no UTC offset")
    return parsed


def validate_record(record: dict, batches: dict[str, dict]) -> dict:
    """Enforce the frozen tier/precision rules; return a normalized VERIFIED row."""
    batch_id = record.get("event_batch_id")
    if batch_id not in batches:
        raise AnnouncementRejection("OFF_CENSUS", f"unknown batch {batch_id!r}")
    precision = record.get("timestamp_precision")
    if precision not in VALID_PRECISION:
        raise AnnouncementRejection("DATE_ONLY",
                                    f"precision {precision!r} is not exact")
    try:
        tier = int(record.get("source_tier"))
    except (TypeError, ValueError):
        tier = None
    if tier not in VALID_TIERS:
        raise AnnouncementRejection("BAD_TIER", f"source_tier {record.get('source_tier')!r}")
    if not record.get("source_url") or not record.get("source_sha256"):
        raise AnnouncementRejection("NO_SOURCE_PROVENANCE",
                                    "source_url and source_sha256 are required")
    kind = record.get("original_or_correction")
    if kind not in VALID_KIND:
        raise AnnouncementRejection("BAD_KIND", f"original_or_correction {kind!r}")
    affected = [t for t in (record.get("affected_tickers") or [])]
    if not affected:
        raise AnnouncementRejection("NO_AFFECTED_TICKERS", "empty affected_tickers")
    unknown = set(affected) - set(batches[batch_id]["tickers"])
    if unknown:
        raise AnnouncementRejection("TICKER_NOT_IN_BATCH", f"{sorted(unknown)}")
    public_time = _parse_time(record.get("announcement_public_time"))
    return {
        "event_batch_id": batch_id,
        "announcement_public_time": public_time.isoformat(),
        "announcement_timezone": record.get("announcement_timezone"),
        "effective_date": batches[batch_id]["effective_date"],
        "source_tier": tier,
        "source_url": record["source_url"],
        "source_sha256": record["source_sha256"],
        "original_or_correction": kind,
        "affected_tickers": sorted(affected),
        "timestamp_precision": precision,
        "first_executable_time": classify_entry(public_time),
        "verification_status": "VERIFIED",
    }


def ingest_resolved(path: str, verbose: bool = True) -> dict:
    """Validate resolved records, freeze the manifest + rejection ledger. No join."""
    con = connect(read_only=True)
    batches = _addition_batches(con)
    con.close()
    with open(path) as handle:
        candidates = [json.loads(line) for line in handle if line.strip()]

    verified, rejections = [], []
    resolved_batches = set()
    for record in candidates:
        try:
            row = validate_record(record, batches)
            verified.append(row)
            resolved_batches.add(row["event_batch_id"])
        except AnnouncementRejection as rej:
            rejections.append({"event_batch_id": record.get("event_batch_id"),
                               "reason_code": rej.reason_code, "detail": rej.detail})
    unresolved = [{"event_batch_id": b, "reason_code": "UNRESOLVED",
                   "detail": "no verified Tier 1/2/3 announcement time sourced"}
                  for b in sorted(batches) if b not in resolved_batches]

    with open(MANIFEST_PATH, "w") as file:
        for row in sorted(verified, key=lambda r: r["announcement_public_time"]):
            file.write(json.dumps(row) + "\n")
    ledger = {"rules_version": RULES_VERSION, "total_batches": len(batches),
              "verified_batches": len(resolved_batches),
              "rejected": rejections, "unresolved": unresolved}
    with open(LEDGER_PATH, "w") as file:
        json.dump(ledger, file, indent=2)

    summary = {"total_batches": len(batches), "verified_records": len(verified),
               "verified_batches": len(resolved_batches),
               "rejected": len(rejections), "unresolved": len(unresolved)}
    if verbose:
        print(f"announcement ingest: {summary['verified_batches']}/{summary['total_batches']} "
              f"batches verified; {summary['rejected']} rejected, "
              f"{summary['unresolved']} unresolved")
    return summary


if __name__ == "__main__":
    generate_worklist()
